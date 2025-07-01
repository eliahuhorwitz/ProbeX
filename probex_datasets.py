import os
import json
import numpy as np
from torch.utils.data import Dataset
from safetensors.torch import safe_open
from utils import tiny_imagenet_id2class

class ModelsDatasetDiscriminative(Dataset):
    def __init__(self, root_dir, label_encoder, layer_name, num_classes, is_resnet):
        self.root_dir = root_dir
        self.file_list = []
        self.label_encoder = label_encoder
        self.layer_name = layer_name
        self.num_classes = num_classes
        self.is_resnet = is_resnet

        for model_dir in os.listdir(root_dir):
            if model_dir.startswith('.DS_Store'):
                continue
            if model_dir.startswith('.locks'):
                continue
            if not os.path.isdir(os.path.join(root_dir, model_dir)):
                continue

            refs_path = os.path.join(root_dir, model_dir, "refs", "main")
            if not os.path.exists(refs_path):
                continue
            with open(refs_path, 'r') as f:
                ref = f.read().strip()

            model_path = os.path.join(root_dir, model_dir, "snapshots", ref, "model.safetensors")
            if os.path.exists(model_path):
                self.file_list.append(model_path)

        with safe_open(self.file_list[0], framework="pt", device="cpu") as f:
            self.metadata = f.metadata()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]

        with safe_open(file_path, framework="pt", device="cpu") as f:
            if self.layer_name in f.keys():
                weight = f.get_tensor(self.layer_name)
                metadata = f.metadata()

        model_chosen_classes = sorted(eval(metadata["dataset_chosen_targets"]))
        model_chosen_classes_ids_in_label_encoder = self.label_encoder.transform(model_chosen_classes)
        assert sorted(self.label_encoder.inverse_transform(model_chosen_classes_ids_in_label_encoder)) == sorted(eval(metadata["dataset_chosen_targets"]))
        y = np.zeros(self.num_classes, dtype=np.float32)
        for class_idx in model_chosen_classes_ids_in_label_encoder:
            y[class_idx] = 1
        y = [y]

        if len(weight.squeeze().shape) == 4 and self.is_resnet:
            weight = weight.squeeze().reshape(-1, weight.squeeze().shape[0]).squeeze()
        elif weight.dim() > 2:
            weight = weight.squeeze()
        elif weight.dim() == 1:
            weight = weight.unsqueeze(1)
        return weight, y[0]


class ModelsDatasetGenerative(Dataset):
    def __init__(self, root_dir, label_encoder):
        self.root_dir = root_dir
        self.file_list = []
        self.label_encoder = label_encoder

        for subset_dir in os.listdir(root_dir):
            if subset_dir.startswith('.DS_Store'):
                continue
            subset_path = os.path.join(root_dir, subset_dir)
            if os.path.isdir(subset_path):
                for file in os.listdir(subset_path):
                    if file.startswith('.DS_Store'):
                        continue
                    if file.endswith('.safetensors'):
                        self.file_list.append(os.path.join(subset_path, file))

        with safe_open(self.file_list[0], framework="pt", device="cpu") as f:
            self.metadata = f.metadata()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]

        with safe_open(file_path, framework="pt", device="cpu") as f:
            B = f.get_tensor("B")
            A = f.get_tensor("A")
            metadata = f.metadata()

        y = [metadata["imagenet_class_id"]]
        y_class = tiny_imagenet_id2class[y[0]]

        y_class = [' '.join(label.split('_')) for label in y_class]

        y_onehot = self.label_encoder.transform(y)
        return B, A, y_class[0], y_onehot[0]
