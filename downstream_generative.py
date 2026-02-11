import os
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.covariance import LedoitWolf
from safetensors.torch import load_model
from torch.utils.data import DataLoader

import faiss
import probex
from probex_datasets import LoRAModelsDatasetGenerative
from utils import tiny_imagenet_id2class, get_class_ids_from_hf_dataset, get_class_ids_from_directory


def extract_features(model, dataloader, device):
    model.eval()
    all_features = []
    all_labels = []
    all_model_ids = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            B, A, class_ids, encoded_labels = batch
            B = B.to(device)
            A = A.to(device)
            inputs = B @ A

            features = model.extract_representation(inputs)

            all_features.append(features.cpu())
            all_labels.extend(class_ids)
            for i, class_id in enumerate(class_ids):
                all_model_ids.append(f"{class_id}_{len(all_model_ids)}")

    features = torch.cat(all_features, dim=0)
    return features, all_labels, all_model_ids


def extract_features_by_class(model, dataloader, device):
    features, labels, model_ids = extract_features(model, dataloader, device)

    class_features = defaultdict(list)
    class_model_ids = defaultdict(list)

    for feat, label, model_id in zip(features, labels, model_ids):
        class_features[label].append(feat)
        class_model_ids[label].append(model_id)

    for class_id in class_features:
        class_features[class_id] = torch.stack(class_features[class_id])

    return dict(class_features), dict(class_model_ids)


def knn_search(train_features, test_features, k=2):
    train_np = train_features.numpy().astype(np.float32)
    test_np = test_features.numpy().astype(np.float32)

    dim = train_np.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(train_np)

    distances, indices = index.search(test_np, k)
    return distances, indices


def knn_anomaly_score(train_features, test_features, k=2):
    distances, _ = knn_search(train_features, test_features, k)
    return np.mean(distances, axis=1)


def run_occ_knn(train_features, test_features, k=2):
    results = {'per_class_auc': {}, 'mean_auc': 0.0}
    aucs = []

    for class_id in tqdm(train_features.keys(), desc="OCC evaluation"):
        normal_train = train_features[class_id]

        normal_test = test_features.get(class_id)
        if normal_test is None or len(normal_test) == 0:
            continue

        anomal_test = torch.cat([test_features[c] for c in test_features if c != class_id], dim=0)

        normal_scores = knn_anomaly_score(normal_train, normal_test, k)
        anomal_scores = knn_anomaly_score(normal_train, anomal_test, k)

        labels = np.concatenate([np.zeros(len(normal_scores)), np.ones(len(anomal_scores))])
        scores = np.concatenate([normal_scores, anomal_scores])

        auc = roc_auc_score(labels, scores)
        aucs.append(auc)
        results['per_class_auc'][class_id] = auc

    results['mean_auc'] = np.mean(aucs)
    return results


def run_occ_ledoit(train_features, test_features):
    results = {'per_class_auc': {}, 'mean_auc': 0.0}
    aucs = []

    for class_id in tqdm(train_features.keys(), desc="OCC (Ledoit-Wolf)"):
        normal_train = train_features[class_id].numpy()

        normal_test = test_features.get(class_id)
        if normal_test is None or len(normal_test) == 0:
            continue
        normal_test = normal_test.numpy()

        anomal_test = torch.cat([test_features[c] for c in test_features if c != class_id], dim=0).numpy()

        lw = LedoitWolf()
        lw.fit(normal_train)

        normal_scores = lw.mahalanobis(normal_test)
        anomal_scores = lw.mahalanobis(anomal_test)

        labels = np.concatenate([np.zeros(len(normal_scores)), np.ones(len(anomal_scores))])
        scores = np.concatenate([normal_scores, anomal_scores])

        auc = roc_auc_score(labels, scores)
        aucs.append(auc)
        results['per_class_auc'][class_id] = auc

    results['mean_auc'] = np.mean(aucs)
    return results


def run_knn_classification_by_class(train_features_dict, test_features_dict, k=1):
    train_features, train_labels = [], []
    for class_id, feats in train_features_dict.items():
        train_features.append(feats)
        train_labels.extend([class_id] * len(feats))
    train_features = torch.cat(train_features, dim=0)

    test_features, test_labels = [], []
    for class_id, feats in test_features_dict.items():
        test_features.append(feats)
        test_labels.extend([class_id] * len(feats))
    test_features = torch.cat(test_features, dim=0)

    distances, indices = knn_search(train_features, test_features, k)

    train_labels_np = np.array(train_labels)
    predictions = []
    for neighbor_indices in indices:
        neighbor_labels = train_labels_np[neighbor_indices]
        unique, counts = np.unique(neighbor_labels, return_counts=True)
        predictions.append(unique[np.argmax(counts)])

    accuracy = accuracy_score(np.array(test_labels), np.array(predictions))
    return {'accuracy': accuracy}


def run_retrieval(features, labels, model_ids, k=7, queries=None):
    features_np = features.numpy().astype(np.float32)

    dim = features_np.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(features_np)

    results = []
    queries_used = set()

    for i, (feat, label, model_id) in enumerate(zip(features, labels, model_ids)):
        if queries is not None:
            if label not in queries or label in queries_used:
                continue
            queries_used.add(label)

        feat_np = feat.numpy().astype(np.float32).reshape(1, -1)
        distances, indices = index.search(feat_np, k + 1)

        neighbor_indices = indices[0][1:]
        neighbor_distances = distances[0][1:]

        result = {
            'query_id': model_id,
            'query_class': label,
            'query_class_name': queries.get(label, label) if queries else label,
        }

        for j, (idx, dist) in enumerate(zip(neighbor_indices, neighbor_distances)):
            result[f'{j+1}nn_id'] = model_ids[idx]
            result[f'{j+1}nn_class'] = labels[idx]
            result[f'{j+1}nn_distance'] = float(dist)

        results.append(result)

    return results


def load_model_and_data(args, device):
    if args.subset:
        print(f"Loading class IDs from HuggingFace: ProbeX/Model-J/{args.subset}")
        train_classes = get_class_ids_from_hf_dataset(args.subset, 'train')
    else:
        print("Scanning directories for class IDs...")
        train_classes = get_class_ids_from_directory(os.path.join(args.input_path, 'train'))

    label_encoder = LabelEncoder()
    label_encoder.fit(sorted(train_classes))

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
        raise ValueError(f"No safetensors files found in {sample_dir}")

    layer_pairs = LoRAModelsDatasetGenerative.get_all_layer_pairs(sample_file)
    layer_name_down, layer_name_up = layer_pairs[args.layer_idx]

    print(f"Using layer {args.layer_idx}: {layer_name_down.split('.lora')[0]}")

    train_dataset = LoRAModelsDatasetGenerative(os.path.join(args.input_path, 'train'), label_encoder, layer_name_down=layer_name_down, layer_name_up=layer_name_up)
    test_dataset = LoRAModelsDatasetGenerative(os.path.join(args.input_path, 'test'), label_encoder, layer_name_down=layer_name_down, layer_name_up=layer_name_up)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=2)

    B, A, _, _ = train_dataset[0]
    input_shape = (B @ A).shape

    model = probex.ProbeXZeroshot(input_shape=input_shape, n_probes=args.n_probes, proj_dim=args.proj_dim, rep_dim=args.rep_dim, clip_dim=args.clip_dim)

    if args.checkpoint_path:
        print(f"Loading checkpoint: {args.checkpoint_path}")
        load_model(model, args.checkpoint_path)

    model = model.to(device)

    return model, train_loader, test_loader, label_encoder


def main():
    parser = argparse.ArgumentParser(description='Downstream evaluation for generative models')

    parser.add_argument('--task', type=str, required=True, choices=['occ', 'occ_ledoit', 'knn', 'retrieval', 'all'], help='Downstream task to run')

    parser.add_argument('--input_path', type=str, required=True, help='Path to the dataset root directory')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to trained ProbeX checkpoint')
    parser.add_argument('--subset', type=str, choices=['SD_1k', 'SD_200'], default=None, help='Dataset subset (loads labels from HuggingFace)')

    parser.add_argument('--layer_idx', type=int, default=0, help='Index of the LoRA layer to use')

    parser.add_argument('--n_probes', type=int, default=128)
    parser.add_argument('--proj_dim', type=int, default=128)
    parser.add_argument('--rep_dim', type=int, default=512)
    parser.add_argument('--clip_dim', type=int, default=768)

    parser.add_argument('--k', type=int, default=2, help='Number of neighbors for kNN')
    parser.add_argument('--batch_size', type=int, default=64)

    parser.add_argument('--output_csv', type=str, default=None, help='Path to save results CSV (for retrieval)')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model, train_loader, test_loader, label_encoder = load_model_and_data(args, device)

    print("\nExtracting ProbeX features...")
    train_by_class, train_model_ids = extract_features_by_class(model, train_loader, device)
    test_by_class, test_model_ids = extract_features_by_class(model, test_loader, device)

    print(f"Train classes: {len(train_by_class)}, Test classes: {len(test_by_class)}")

    if args.task in ['occ', 'all']:
        print("\n" + "=" * 50)
        print("Running OCC (kNN)...")
        occ_results = run_occ_knn(train_by_class, test_by_class, k=args.k)
        print(f"Mean AUC: {occ_results['mean_auc']:.4f}")

    if args.task in ['occ_ledoit', 'all']:
        print("\n" + "=" * 50)
        print("Running OCC (Ledoit-Wolf)...")
        occ_lw_results = run_occ_ledoit(train_by_class, test_by_class)
        print(f"Mean AUC: {occ_lw_results['mean_auc']:.4f}")

    if args.task in ['knn', 'all']:
        print("\n" + "=" * 50)
        print("Running kNN Classification...")
        knn_results = run_knn_classification_by_class(train_by_class, test_by_class, k=args.k)
        print(f"Accuracy: {knn_results['accuracy'] * 100:.2f}%")

    if args.task in ['retrieval', 'all']:
        print("\n" + "=" * 50)
        print("Running Retrieval...")

        all_features = []
        all_labels = []
        all_model_ids = []

        for class_id, feats in train_by_class.items():
            all_features.append(feats)
            for i in range(len(feats)):
                all_labels.append(class_id)
                all_model_ids.append(f"{class_id}_train_{i}")

        for class_id, feats in test_by_class.items():
            all_features.append(feats)
            for i in range(len(feats)):
                all_labels.append(class_id)
                all_model_ids.append(f"{class_id}_test_{i}")

        all_features = torch.cat(all_features, dim=0)

        queries = {}
        for class_id in set(all_labels):
            if class_id in tiny_imagenet_id2class:
                queries[class_id] = tiny_imagenet_id2class[class_id].replace('_', ' ')
            else:
                queries[class_id] = class_id

        retrieval_results = run_retrieval(all_features, all_labels, all_model_ids, k=args.k, queries=queries)

        if args.output_csv:
            df = pd.DataFrame(retrieval_results)
            df.to_csv(args.output_csv, index=False)
            print(f"Retrieval results saved to: {args.output_csv}")
        else:
            for result in retrieval_results[:5]:
                print(f"\nQuery: {result['query_class_name']} ({result['query_id']})")
                for j in range(min(3, args.k)):
                    print(f"  {j+1}NN: {result.get(f'{j+1}nn_class', 'N/A')}")

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
