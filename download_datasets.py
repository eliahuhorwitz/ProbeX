from datasets import load_dataset
import argparse
from huggingface_hub import snapshot_download, cached_assets_path
from huggingface_hub import cached_assets_path
import os
import requests
from tqdm import tqdm




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract all layers from the tensors')
    parser.add_argument('--dataset_subset', type=str, help='The dimension of the final representation', choices=["SupViT", "DINO", "MAE", "ResNet"], required=True)
    args = parser.parse_args()

    assets_path = cached_assets_path(library_name="ProbeX", namespace="ModelJ")
    download_dir = os.path.join(assets_path, "models", args.dataset_subset)
    os.makedirs(download_dir, exist_ok=True)

    ds = load_dataset("ProbeX/Model-J", args.dataset_subset)

    for split in ["train", "val", "test"]:
        tempt_download_dir = os.path.join(download_dir, split)
        for idx, example in tqdm(enumerate(ds[split])):
            model_url = example["hf_model_url"]
            model_id = example["hf_model_id"]
            cache_path = os.path.join(tempt_download_dir, model_id)

            # Note: Check if the model is already in the cache
            if os.path.exists(cache_path):
                downloaded_path = cache_path
            else:
                try:
                    response = requests.head(model_url, allow_redirects=True)
                    if response.status_code != 200:
                        print(f"URL not accessible: {model_url} (status: {response.status_code})")
                        downloaded_path = None
                        continue
                except Exception as e:
                    print(f"Failed to validate URL: {model_url}. Error: {e}")
                    downloaded_path = None
                    continue

                # Note: Download the model from Hugging Face
                try:
                    downloaded_path = snapshot_download(repo_id=model_id, cache_dir=tempt_download_dir, )
                except Exception as e:
                    print(f"Failed to download {model_id} from {model_url}: {e}")
                    downloaded_path = None  # Handle download failures gracefully
                print(f"Downloaded model into: {str(downloaded_path)}/model.safetensors")
    new_assets_path = cached_assets_path(library_name="ProbeX", namespace="ModelJ")
