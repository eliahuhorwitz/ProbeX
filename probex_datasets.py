import os
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


class LoRAModelsDatasetGenerative(Dataset):
    """
    :param root_dir: Root directory containing the LoRA safetensors files.
    :param label_encoder: Sklearn LabelEncoder fitted on the class IDs.
    :param layer_name_down: The name of the 'down' (A) weight matrix to extract.
    :param layer_name_up: The name of the 'up' (B) weight matrix to extract.
    """

    def __init__(self, root_dir, label_encoder, layer_name_down, layer_name_up,
                 safetensors_filename='pytorch_lora_weights.safetensors'):
        self.root_dir = root_dir
        self.file_list = []
        self.label_encoder = label_encoder
        self.layer_name_down = layer_name_down
        self.layer_name_up = layer_name_up
        self.safetensors_filename = safetensors_filename

        self._find_safetensors_files(root_dir)
        assert len(self.file_list) > 0, f"No safetensors files found in {root_dir}"

        with safe_open(self.file_list[0], framework="pt", device="cpu") as f:
            self.metadata = f.metadata()
            available_keys = list(f.keys())

        assert self.layer_name_down in available_keys, f"Layer '{self.layer_name_down}' not found."
        assert self.layer_name_up in available_keys, f"Layer '{self.layer_name_up}' not found."

    def _find_safetensors_files(self, directory):
        for item in os.listdir(directory):
            if item.startswith('.'):
                continue
            item_path = os.path.join(directory, item)
            if os.path.isdir(item_path):
                safetensors_path = os.path.join(item_path, self.safetensors_filename)
                if os.path.exists(safetensors_path):
                    self.file_list.append(safetensors_path)
                else:
                    self._find_safetensors_files(item_path)
            elif item == self.safetensors_filename:
                self.file_list.append(item_path)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]

        with safe_open(file_path, framework="pt", device="cpu") as f:
            A = f.get_tensor(self.layer_name_down)
            B = f.get_tensor(self.layer_name_up)
            metadata = f.metadata()

        imagenet_class_id = metadata["imagenet_class_id"]
        y_class_names = tiny_imagenet_id2class[imagenet_class_id]
        y_class_text = ' '.join(y_class_names.split('_'))
        y_label = self.label_encoder.transform([imagenet_class_id])[0]

        return B, A, y_class_text, y_label

    def get_layer_shape(self):
        with safe_open(self.file_list[0], framework="pt", device="cpu") as f:
            A = f.get_tensor(self.layer_name_down)
            B = f.get_tensor(self.layer_name_up)
        return (B.shape[0], A.shape[1])

    @classmethod
    def get_all_layer_pairs(cls, safetensors_path):
        with safe_open(safetensors_path, framework="pt", device="cpu") as f:
            keys = list(f.keys())

        down_layers = [k for k in keys if '.lora.down.weight' in k]
        layer_pairs = []

        for down_layer in down_layers:
            up_layer = down_layer.replace('.lora.down.weight', '.lora.up.weight')
            if up_layer in keys:
                layer_pairs.append((down_layer, up_layer))

        return layer_pairs
