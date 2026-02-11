import os
import json
import torch
import probex
import argparse
import shortuuid
import pandas as pd
import clip
from torch import nn
from tqdm import tqdm
from torch.optim import Adam
from datetime import datetime
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from safetensors.torch import load_model, save_file
from probex_datasets import LoRAModelsDatasetGenerative
from utils import fix_seeds, imagenet_ids_to_class_names, get_class_ids_from_hf_dataset, get_class_ids_from_directory


def get_clip_features(labels_dict, clip_model, device):
    class_ids = []
    all_tokens = []

    for class_id, class_name in labels_dict.items():
        tokens = clip.tokenize(f"a photo of a {class_name}").to(device)
        all_tokens.append(tokens)
        class_ids.append(class_id)

    tokenized = torch.cat(all_tokens)
    with torch.no_grad():
        class_features = clip_model.encode_text(tokenized).float()

    return class_ids, class_features


def compute_accuracy(logits, labels):
    similarity = 100.0 * logits
    pred = similarity.topk(1, 1, True, True)[1].t()
    correct = pred.eq(labels.view(1, -1).expand_as(pred))
    acc = float(correct[:1].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    return acc


def setup_training(args, device, clip_model):
    if args.subset:
        print(f"Loading class IDs from HuggingFace dataset: ProbeX/Model-J/{args.subset}")
        train_labels = get_class_ids_from_hf_dataset(args.subset, 'train')
        val_holdout_labels = get_class_ids_from_hf_dataset(args.subset, 'val_holdout')
        test_holdout_labels = get_class_ids_from_hf_dataset(args.subset, 'test_holdout')
    else:
        print("No subset specified, scanning directories for class IDs...")
        train_labels = get_class_ids_from_directory(os.path.join(args.input_path, 'train'))
        val_holdout_labels = get_class_ids_from_directory(os.path.join(args.input_path, 'val_holdout'))
        test_holdout_labels = get_class_ids_from_directory(os.path.join(args.input_path, 'test_holdout'))

    train_labels = list(set(train_labels))
    holdout_labels = list(set(val_holdout_labels) | set(test_holdout_labels))

    print(f"Number of training classes: {len(train_labels)}")
    print(f"Number of holdout classes: {len(holdout_labels)}")

    train_label_encoder = LabelEncoder()
    train_label_encoder.fit(sorted(train_labels))

    holdout_label_encoder = LabelEncoder()
    holdout_label_encoder.fit(sorted(holdout_labels))

    train_labels_dict = imagenet_ids_to_class_names(sorted(train_labels))
    holdout_labels_dict = imagenet_ids_to_class_names(sorted(holdout_labels))

    _, train_clip_features = get_clip_features(train_labels_dict, clip_model, device)
    _, holdout_clip_features = get_clip_features(holdout_labels_dict, clip_model, device)

    layer_name_down, layer_name_up = args.layer_name_down, args.layer_name_up

    print(f"\nCreating datasets for layer: {layer_name_down.split('.')[-3]}")

    train_dataset = LoRAModelsDatasetGenerative(os.path.join(args.input_path, 'train'), train_label_encoder, layer_name_down=layer_name_down, layer_name_up=layer_name_up)
    val_dataset = LoRAModelsDatasetGenerative(os.path.join(args.input_path, 'val'), train_label_encoder, layer_name_down=layer_name_down, layer_name_up=layer_name_up)
    test_dataset = LoRAModelsDatasetGenerative(os.path.join(args.input_path, 'test'), train_label_encoder, layer_name_down=layer_name_down, layer_name_up=layer_name_up)
    val_holdout_dataset = LoRAModelsDatasetGenerative(os.path.join(args.input_path, 'val_holdout'), holdout_label_encoder, layer_name_down=layer_name_down, layer_name_up=layer_name_up)
    test_holdout_dataset = LoRAModelsDatasetGenerative(os.path.join(args.input_path, 'test_holdout'), holdout_label_encoder, layer_name_down=layer_name_down, layer_name_up=layer_name_up)

    print(f"Dataset sizes: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}, val_holdout={len(val_holdout_dataset)}, test_holdout={len(test_holdout_dataset)}")

    num_workers = 2
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=num_workers)
    val_holdout_loader = DataLoader(val_holdout_dataset, batch_size=args.batch_size, num_workers=num_workers)
    test_holdout_loader = DataLoader(test_holdout_dataset, batch_size=args.batch_size, num_workers=num_workers)

    loaders_dict = {
        'train': train_loader, 'val': val_loader, 'test': test_loader,
        'val_holdout': val_holdout_loader, 'test_holdout': test_holdout_loader
    }

    clip_features_dict = {'train': train_clip_features, 'holdout': holdout_clip_features}

    B, A, _, _ = train_dataset[0]
    input_shape = (B @ A).shape
    print(f"Input shape (B @ A): {input_shape}")

    model = probex.ProbeXZeroshot(input_shape=input_shape, n_probes=args.n_probes, proj_dim=args.proj_dim, rep_dim=args.rep_dim, clip_dim=args.clip_dim)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.training_lr, betas=(0.9, 0.999))

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTraining total number of trainable parameters: {n_trainable}/{n_params}")

    return model, loaders_dict, clip_features_dict, criterion, optimizer, n_params


def run_eval(model, loader, clip_features, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for B, A, y_text, y_label in loader:
            B = B.to(device)
            A = A.to(device)
            inputs = B @ A
            y_label = y_label.to(device)

            logits = model(inputs, clip_features)
            loss = criterion(logits, y_label)

            total_loss += loss.item()
            total_samples += B.size(0)
            total_correct += compute_accuracy(logits, y_label)

    avg_loss = total_loss / len(loader)
    accuracy = 100 * total_correct / total_samples
    return avg_loss, accuracy


def run_train(args, model, loaders, clip_features, criterion, optimizer, device):
    train_loader = loaders['train']
    val_loader = loaders['val']
    val_holdout_loader = loaders['val_holdout']

    train_clip_features = clip_features['train']
    holdout_clip_features = clip_features['holdout']

    best = {
        'val_accuracy': 0.0, 'val_loss': float('inf'), 'val_epoch': 0,
        'val_holdout_accuracy': 0.0, 'val_holdout_loss': float('inf'), 'val_holdout_epoch': 0,
    }

    for epoch in tqdm(range(args.n_epochs)):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for B, A, y_text, y_label in train_loader:
            B = B.to(device)
            A = A.to(device)
            inputs = B @ A
            y_label = y_label.to(device)

            optimizer.zero_grad()
            logits = model(inputs, train_clip_features)
            loss = criterion(logits, y_label)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_total += B.size(0)
            train_correct += compute_accuracy(logits, y_label)

        train_loss /= len(train_loader)
        train_accuracy = 100 * train_correct / train_total

        if epoch % args.eval_every == 0:
            print(f"\nEpoch [{epoch + 1}/{args.n_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%")

            val_loss, val_accuracy = run_eval(model, val_loader, train_clip_features, criterion, device)
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

            if val_accuracy >= best['val_accuracy']:
                best['val_accuracy'] = val_accuracy
                best['val_loss'] = val_loss
                best['val_epoch'] = epoch
                best['train_loss'] = train_loss
                best['train_accuracy'] = train_accuracy
                save_file(model.state_dict(), args.checkpoint_path_val, metadata={"epoch": str(epoch)})

            val_holdout_loss, val_holdout_accuracy = run_eval(model, val_holdout_loader, holdout_clip_features, criterion, device)
            print(f"  Val Holdout Loss: {val_holdout_loss:.4f}, Val Holdout Acc: {val_holdout_accuracy:.2f}%")

            if val_holdout_accuracy >= best['val_holdout_accuracy']:
                best['val_holdout_accuracy'] = val_holdout_accuracy
                best['val_holdout_loss'] = val_holdout_loss
                best['val_holdout_epoch'] = epoch
                save_file(model.state_dict(), args.checkpoint_path_val_holdout, metadata={"epoch": str(epoch)})

    return best


def final_evaluation(args, model, loaders, clip_features, criterion, device):
    train_loader = loaders['train']
    val_loader = loaders['val']
    test_loader = loaders['test']
    val_holdout_loader = loaders['val_holdout']
    test_holdout_loader = loaders['test_holdout']

    train_clip_features = clip_features['train']
    holdout_clip_features = clip_features['holdout']

    results = {}

    # Evaluate best val checkpoint
    print("\n" + "=" * 50)
    print("Evaluating best val checkpoint:")
    load_model(model, args.checkpoint_path_val)
    model = model.to(device)

    _, results['best_val_train_acc'] = run_eval(model, train_loader, train_clip_features, criterion, device)
    _, results['best_val_val_acc'] = run_eval(model, val_loader, train_clip_features, criterion, device)
    _, results['best_val_test_acc'] = run_eval(model, test_loader, train_clip_features, criterion, device)
    _, results['best_val_val_holdout_acc'] = run_eval(model, val_holdout_loader, holdout_clip_features, criterion, device)
    _, results['best_val_test_holdout_acc'] = run_eval(model, test_holdout_loader, holdout_clip_features, criterion, device)

    print(f"  Train: {results['best_val_train_acc']:.2f}%, Val: {results['best_val_val_acc']:.2f}%, Test: {results['best_val_test_acc']:.2f}%")
    print(f"  Val Holdout: {results['best_val_val_holdout_acc']:.2f}%, Test Holdout: {results['best_val_test_holdout_acc']:.2f}%")

    # Evaluate best val_holdout checkpoint
    print("\nEvaluating best val_holdout checkpoint:")
    load_model(model, args.checkpoint_path_val_holdout)
    model = model.to(device)

    _, results['best_holdout_train_acc'] = run_eval(model, train_loader, train_clip_features, criterion, device)
    _, results['best_holdout_val_acc'] = run_eval(model, val_loader, train_clip_features, criterion, device)
    _, results['best_holdout_test_acc'] = run_eval(model, test_loader, train_clip_features, criterion, device)
    _, results['best_holdout_val_holdout_acc'] = run_eval(model, val_holdout_loader, holdout_clip_features, criterion, device)
    _, results['best_holdout_test_holdout_acc'] = run_eval(model, test_holdout_loader, holdout_clip_features, criterion, device)

    print(f"  Train: {results['best_holdout_train_acc']:.2f}%, Val: {results['best_holdout_val_acc']:.2f}%, Test: {results['best_holdout_test_acc']:.2f}%")
    print(f"  Val Holdout: {results['best_holdout_val_holdout_acc']:.2f}%, Test Holdout: {results['best_holdout_test_holdout_acc']:.2f}%")

    return results


def train_layer(args, device, clip_model):
    model, loaders, clip_features, criterion, optimizer, n_params = setup_training(args, device, clip_model)
    best_metrics = run_train(args, model, loaders, clip_features, criterion, optimizer, device)
    final_results = final_evaluation(args, model, loaders, clip_features, criterion, device)

    results = {
        'layer_idx': args.current_layer,
        'layer_name': args.layer_name_down.split('.lora')[0],
        'n_params': n_params,
        'best_val_epoch': best_metrics['val_epoch'],
        'best_val_holdout_epoch': best_metrics['val_holdout_epoch'],
        **final_results
    }

    return results


def default_argument_parser():
    parser = argparse.ArgumentParser(description='Train ProbeX for generative (LoRA) models')

    parser.add_argument('--input_path', type=str, required=True, help='Path to the dataset root directory')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save results and checkpoints')
    parser.add_argument('--subset', type=str, choices=['SD_1k', 'SD_200'], default=None, help='Dataset subset (loads labels from HuggingFace)')

    parser.add_argument('--start_layer', type=int, default=0, help='The index of the first layer to classify')
    parser.add_argument('--n_layers', type=int, default=1, help='The number of layers to classify')

    parser.add_argument('--n_probes', type=int, default=128, help='The number of probes to use')
    parser.add_argument('--proj_dim', type=int, default=128, help='The dimension to project the weight matrix to')
    parser.add_argument('--rep_dim', type=int, default=512, help='The dimension of the final representation')
    parser.add_argument('--clip_dim', type=int, default=768, help='CLIP embedding dimension (768 for ViT-L/14)')
    parser.add_argument('--clip_model', type=str, default='ViT-L/14', help='CLIP model to use')

    parser.add_argument('--training_lr', type=float, default=0.002, help='The learning rate for the model')
    parser.add_argument('--batch_size', type=int, default=128, help='The batch size for the model')
    parser.add_argument('--n_epochs', type=int, default=200, help='The number of epochs for the model')
    parser.add_argument('--eval_every', type=int, default=10, help='Evaluate every N epochs')

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--suffix', type=str, default='', help='The suffix to use')

    return parser


def setup_paths(args):
    args.output_path_root = f"{args.output_path}/ProbeX/n_probes-{args.n_probes}/proj_dim-{args.proj_dim}/rep_dim-{args.rep_dim}"
    os.makedirs(args.output_path_root, exist_ok=True)
    suffix = args.suffix if args.suffix else shortuuid.ShortUUID().random(length=5)
    args.final_suffix = f"{suffix}_{datetime.now().strftime('%H-%M-%d-%m')}"
    os.makedirs(os.path.join(args.output_path_root, "args"), exist_ok=True)
    os.makedirs(os.path.join(args.output_path_root, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(args.output_path_root, "results"), exist_ok=True)


def main():
    args = default_argument_parser().parse_args()
    setup_paths(args)
    fix_seeds(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print(f"Loading CLIP model: {args.clip_model}")
    clip_model, _ = clip.load(args.clip_model, device=device)

    # Get a sample safetensors file to determine available layers
    sample_dir = os.path.join(args.input_path, 'train')
    sample_file = None
    for root, dirs, files in os.walk(sample_dir):
        for f in files:
            if f == 'pytorch_lora_weights.safetensors':
                sample_file = os.path.join(root, f)
                break
        if sample_file:
            break

    if sample_file is None:
        raise ValueError(f"No pytorch_lora_weights.safetensors found in {sample_dir}")

    all_layer_pairs = LoRAModelsDatasetGenerative.get_all_layer_pairs(sample_file)
    print(f"\nFound {len(all_layer_pairs)} layer pairs in LoRA model")

    print(f"\nRunning with arguments:\n{json.dumps(vars(args), indent=4)}")

    all_results = []

    for layer_idx in tqdm(range(args.start_layer, args.start_layer + args.n_layers)):
        args.current_layer = layer_idx
        args.layer_name_down, args.layer_name_up = all_layer_pairs[layer_idx]

        args.checkpoint_path_val = os.path.join(
            args.output_path_root, "checkpoints",
            f"best_val_layer-{layer_idx}__{args.final_suffix}.safetensors"
        )
        args.checkpoint_path_val_holdout = os.path.join(
            args.output_path_root, "checkpoints",
            f"best_val_holdout_layer-{layer_idx}__{args.final_suffix}.safetensors"
        )

        with open(os.path.join(args.output_path_root, "args", f"args_layer-{layer_idx}__{args.final_suffix}.json"), "w") as f:
            json.dump(vars(args), f, indent=4)

        print("\n\n\n" + "=" * 25 + " NEW LAYER " + "=" * 25)
        print(f"Starting to run on layer {layer_idx} (layer name: {args.layer_name_down.split('.lora')[0]})")

        results = train_layer(args, device, clip_model)
        all_results.append(results)

        results_df = pd.DataFrame([results])
        results_path = os.path.join(args.output_path_root, "results", f"results_layer-{layer_idx}__{args.final_suffix}.csv")
        results_df.to_csv(results_path, index=False)
        print(f"Finished layer {layer_idx}, results saved to {results_path}")

    if len(all_results) > 1:
        combined_df = pd.DataFrame(all_results)
        combined_path = os.path.join(
            args.output_path_root, "results",
            f"results_layers-{args.start_layer}-{args.start_layer + args.n_layers}__{args.final_suffix}.csv"
        )
        combined_df.to_csv(combined_path, index=False)
        print(f"\nCombined results saved to: {combined_path}")


if __name__ == "__main__":
    main()
