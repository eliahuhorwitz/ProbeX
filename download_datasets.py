from datasets import load_dataset
import argparse
import time
from huggingface_hub import snapshot_download, cached_assets_path, HfApi, hf_hub_download
import os
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


DISCRIMINATIVE_SUBSETS = ["SupViT", "DINO", "MAE", "ResNet"]
DISCRIMINATIVE_SPLITS = ["train", "val", "test"]

GENERATIVE_SUBSETS = ["SD_1k", "SD_200"]
GENERATIVE_SPLITS = ["train", "val", "test", "val_holdout", "test_holdout"]


def _count_models_in_dir(path, splits):
    total = 0
    for split in splits:
        split_dir = os.path.join(path, split)
        if os.path.isdir(split_dir):
            total += len([d for d in os.listdir(split_dir)
                         if os.path.isdir(os.path.join(split_dir, d)) and not d.startswith('.')])
    return total


def _download_single_file(repo_id, filename, local_dir, force, max_retries=5):
    local_path = os.path.join(local_dir, filename)

    if not force and os.path.exists(local_path):
        return filename, "skipped"

    for attempt in range(max_retries):
        try:
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="model",
                local_dir=local_dir,
                force_download=force,
            )
            return filename, "ok"
        except Exception as e:
            if attempt < max_retries - 1:
                wait = min(2 ** attempt, 30)
                time.sleep(wait)
            else:
                return filename, f"error: {e}"
    return filename, "error: max retries exceeded"


def _get_model_image_files(api, repo_id, model_dir, max_retries=5):
    images_dir = f"{model_dir}/training_images"
    for attempt in range(max_retries):
        try:
            files = []
            for item in api.list_repo_tree(
                repo_id=repo_id,
                path_in_repo=images_dir,
                recursive=False,
                repo_type="model",
            ):
                if hasattr(item, 'rfilename'):
                    files.append(item.rfilename)
            return files
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return []


def download_generative_models(dataset_subset, splits=None, output_dir=None,
                                force_download=False, include_images=False):
    if splits is None:
        splits = GENERATIVE_SPLITS

    repo_id = f"ProbeX/Model-J__{dataset_subset}"

    if output_dir:
        local_dir = os.path.join(output_dir, dataset_subset)
    else:
        assets_path = cached_assets_path(library_name="ProbeX", namespace="ModelJ")
        local_dir = os.path.join(assets_path, "models", dataset_subset)

    os.makedirs(local_dir, exist_ok=True)

    print(f"Downloading {dataset_subset} from: {repo_id}")
    print(f"  Destination: {local_dir}")
    print(f"  Splits: {splits}")
    if force_download:
        print(f"  Force download: enabled")
    else:
        print(f"  Resumable: existing files will be skipped")

    api = HfApi()
    try:
        repo_info = api.repo_info(repo_id=repo_id, repo_type="model")
        print(f"  Repo verified: {repo_id} (revision: {repo_info.sha[:8]})")
    except Exception as e:
        print(f"\nERROR: Could not access repo {repo_id}: {e}")
        return local_dir

    print(f"\n  Loading model metadata from ProbeX/Model-J/{dataset_subset}...")
    ds = load_dataset("ProbeX/Model-J", dataset_subset, trust_remote_code=True)

    split_files = {}
    for split in splits:
        if split not in ds:
            print(f"  WARNING: Split '{split}' not found in dataset, skipping...")
            continue
        paths = []
        for row in ds[split]:
            if 'hf_model_path' in row and row['hf_model_path']:
                paths.append(row['hf_model_path'])
            else:
                idx = int(row['model_idx'])
                paths.append(f"{split}/model_idx_{idx:04d}/pytorch_lora_weights.safetensors")
        split_files[split] = paths

    total_models = sum(len(v) for v in split_files.values())
    print(f"  Found {total_models} models across {len(split_files)} splits")

    existing_models = _count_models_in_dir(local_dir, splits)
    if existing_models > 0:
        print(f"  Found {existing_models} existing models locally")

    grand_downloaded = 0
    grand_skipped = 0
    grand_errors = 0

    for split, safetensor_paths in split_files.items():
        print(f"\n  [{split}] {len(safetensor_paths)} models")

        files_to_download = list(safetensor_paths)

        if include_images:
            print(f"  [{split}] Discovering training images...")
            model_dirs = set()
            for p in safetensor_paths:
                model_dirs.add(os.path.dirname(p))

            for model_dir in tqdm(sorted(model_dirs), desc=f"  [{split}] Listing images", unit="model"):
                image_files = _get_model_image_files(api, repo_id, model_dir)
                files_to_download.extend(image_files)

            print(f"  [{split}] Total files (weights + images): {len(files_to_download)}")

        to_download = []
        skipped = 0
        for f in files_to_download:
            local_path = os.path.join(local_dir, f)
            if not force_download and os.path.exists(local_path):
                skipped += 1
            else:
                to_download.append(f)

        if skipped > 0:
            print(f"  [{split}] Skipping {skipped} already-downloaded files")

        if not to_download:
            print(f"  [{split}] All files already downloaded!")
            grand_skipped += skipped
            continue

        print(f"  [{split}] Downloading {len(to_download)} files...")

        downloaded = 0
        errors = []

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(_download_single_file, repo_id, f, local_dir, force_download): f
                for f in to_download
            }

            with tqdm(total=len(to_download), desc=f"  [{split}]", unit="file") as pbar:
                for future in as_completed(futures):
                    filename, status = future.result()
                    if status == "ok":
                        downloaded += 1
                    elif status == "skipped":
                        skipped += 1
                    else:
                        errors.append((filename, status))
                    pbar.update(1)

        if errors:
            print(f"  [{split}] WARNING: {len(errors)} files failed:")
            for fname, err in errors[:5]:
                print(f"    {fname}: {err}")
            if len(errors) > 5:
                print(f"    ... and {len(errors) - 5} more")

        grand_downloaded += downloaded
        grand_skipped += skipped
        grand_errors += len(errors)

    print(f"\nDownload complete! Models saved to: {local_dir}")
    print(f"  Files: {grand_downloaded} downloaded, {grand_skipped} skipped, {grand_errors} errors")

    for split in splits:
        split_dir = os.path.join(local_dir, split)
        if os.path.isdir(split_dir):
            n_models = len([d for d in os.listdir(split_dir)
                          if os.path.isdir(os.path.join(split_dir, d)) and not d.startswith('.')])
            print(f"  {split}: {n_models} models")
        else:
            print(f"  {split}: not found")

    if grand_errors > 0:
        print(f"\n  {grand_errors} files failed. Re-run the same command to retry — "
              f"already-downloaded files will be skipped.")

    return local_dir


def download_discriminative_models(dataset_subset, splits=None, output_dir=None, force_download=False):
    if splits is None:
        splits = DISCRIMINATIVE_SPLITS

    if output_dir:
        download_dir = os.path.join(output_dir, dataset_subset)
    else:
        assets_path = cached_assets_path(library_name="ProbeX", namespace="ModelJ")
        download_dir = os.path.join(assets_path, "models", dataset_subset)
    os.makedirs(download_dir, exist_ok=True)

    print(f"Downloading {dataset_subset} to: {download_dir}")
    print(f"Splits: {splits}")

    ds = load_dataset("ProbeX/Model-J", dataset_subset)

    for split in splits:
        if split not in ds:
            print(f"Warning: Split '{split}' not found in dataset, skipping...")
            continue

        split_download_dir = os.path.join(download_dir, split)
        os.makedirs(split_download_dir, exist_ok=True)

        print(f"\nDownloading {split} split ({len(ds[split])} models)...")

        for idx, example in tqdm(enumerate(ds[split]), total=len(ds[split])):
            model_url = example["hf_model_url"]
            model_id = example["hf_model_id"]
            cache_path = os.path.join(split_download_dir, model_id.replace("/", "__"))

            if os.path.exists(cache_path) and not force_download:
                continue

            try:
                response = requests.head(model_url, allow_redirects=True)
                if response.status_code != 200:
                    print(f"URL not accessible: {model_url} (status: {response.status_code})")
                    continue
            except Exception as e:
                print(f"Failed to validate URL: {model_url}. Error: {e}")
                continue

            try:
                downloaded_path = snapshot_download(
                    repo_id=model_id,
                    cache_dir=split_download_dir,
                    force_download=force_download,
                )
                print(f"Downloaded: {model_id}")
            except Exception as e:
                print(f"Failed to download {model_id}: {e}")

    print(f"\nDownload complete! Models saved to: {download_dir}")
    return download_dir


def download_models(dataset_subset, splits=None, output_dir=None,
                    force_download=False, include_images=False):
    if dataset_subset in GENERATIVE_SUBSETS:
        return download_generative_models(dataset_subset, splits, output_dir,
                                          force_download, include_images)
    else:
        return download_discriminative_models(dataset_subset, splits, output_dir, force_download)


if __name__ == "__main__":
    all_subsets = DISCRIMINATIVE_SUBSETS + GENERATIVE_SUBSETS

    parser = argparse.ArgumentParser(description='Download ProbeX Model-J datasets from HuggingFace')
    parser.add_argument('--dataset_subset', type=str, required=True, choices=all_subsets, help='Dataset subset to download')
    parser.add_argument('--splits', type=str, nargs='+', default=None, help='Specific splits to download (default: all)')
    parser.add_argument('--output_dir', type=str, default=None, help='Custom output directory (default: HF cached assets)')
    parser.add_argument('--force_download', action='store_true', help='Force re-download all files')
    parser.add_argument('--include_images', action='store_true', help='Also download training images (generative subsets only)')

    args = parser.parse_args()
    download_models(args.dataset_subset, args.splits, args.output_dir,
                    args.force_download, args.include_images)
