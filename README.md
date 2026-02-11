# Learning on Model Weights using Tree Experts
Official PyTorch Implementation for the "Learning on Model Weights using Tree Experts" paper (CVPR 2025).  
<p align="center">
    🌐 <a href="https://horwitz.ai/probex" target="_blank">Project</a> | 📃 <a href="https://arxiv.org/abs/2410.13569" target="_blank">Paper</a> | 🤗 <a href="https://huggingface.co/ProbeX" target="_blank">Dataset</a> <br>
</p>

![](imgs/poster.png)

___

> **Learning on Model Weights using Tree Experts**<br>
> Eliahu Horwitz*, Bar Cavia*, Jonathan Kahana*, Yedid Hoshen<br>
> <a href="https://arxiv.org/abs/2410.13569" target="_blank">https://arxiv.org/abs/2410.13569 </a> <br>
>
>**Abstract:** The number of publicly available models is rapidly increasing, yet most remain undocumented. 
> Users looking for suitable models for their tasks must first determine what each model does. 
> Training machine learning models to infer missing documentation directly from model weights is challenging, 
> as these weights often contain significant variation unrelated to model functionality (denoted nuisance). 
> Here, we identify a key property of real-world models: most public models belong to a small set of *Model Trees*, 
> where all models within a tree are fine-tuned from a common ancestor (e.g., a foundation model). 
> Importantly, we find that within each tree there is less nuisance variation between models. 
> Concretely, while learning across Model Trees requires complex architectures, even a linear classifier trained 
> on a single model layer often works within trees. While effective, these linear classifiers are computationally 
> expensive, especially when dealing with larger models that have many parameters. To address this, we introduce
> *Probing Experts* (ProbeX), a theoretically motivated and lightweight method. Notably, ProbeX is the first 
> probing method specifically designed to learn from the weights of a single hidden model layer.
> We demonstrate the effectiveness of ProbeX by predicting the categories in a model's training dataset based only
> on its weights. Excitingly, ProbeX can map the weights of Stable Diffusion into a weight-language embedding
> space, enabling model search via text, i.e., zero-shot model classification.


___



## Project Structure
This project consists of:
- `download_datasets.py` - script for downloading the Model-J dataset from Hugging Face.
- `train_discriminative_probex.py` - training ProbeX on the discriminative splits of the Model-J dataset.
- `train_generative_probex.py` - training ProbeX on the generative (LoRA) splits using CLIP-based zero-shot classification.
- `downstream_generative.py` - downstream evaluation tasks (OCC, kNN, retrieval) for generative models.



## Installation 
1.  Clone the repo:
```bash
git clone https://github.com/eliahuhorwitz/ProbeX.git
cd ProbeX
```
2. Create a new environment and install the libraries:
```bash
python3 -m venv ProbeX_venv
source ProbeX_venv/bin/activate
pip install -r requirements.txt
```

## Download the Model-J dataset
The Model-J dataset contains different subsets of discriminative and generative models, to download a subset of models run:
```bash
python download_datasets.py --dataset_subset=SUBSET_NAME
```
Where `SUBSET_NAME` can be one of the following:
- Discriminative: "SupViT", "DINO", "MAE", "ResNet"
- Generative: "SD_1k", "SD_200"

Each discriminative model is stored as a separate Hugging Face model repository. The generative models (SD_1k, SD_200) are each stored as a single Hugging Face model repository containing all LoRA weights. The download script handles both formats automatically, with built-in retry logic and resumable downloads — re-running the same command will skip already-downloaded files.

Each of the discriminative subsets is about 350GB and each generative subset is about 70GB, so we recommend starting with a single subset.

All model hyperparameters are available both as metadata inside each safetensor file and on the [Model-J dataset page](https://huggingface.co/datasets/ProbeX/Model-J).

## Running ProbeX
Below are examples for running ProbeX on the Model-J dataset subsets.   

### Discriminative Splits
ProbeX can be trained on the discriminative splits of the Model-J dataset to predict the categories in a model's training dataset. The training is done using the `train_discriminative_probex.py` script.
ProbeX trains an individual model per layer, to choose the best layer we use the validation set. 
The training script trains layers sequentially, the script supports specifying the start layer and the number of layers to train, allowing this to be distributed across multiple GPUs.

Below is an example of training a single layer, see `sbatch_run_discriminative_probex.sh` for an example of distributing the training of multiple layers across multiple GPUs.

#### ViT
```bash
python train_discriminative_probex.py --input_path="~/.cache/huggingface/assets/ProbeX/ModelJ/default/models/SupViT/" 
--output_path="ProbeX_outputs/SupViT" --is_resnet="False" --start_layer=59 --n_layers=1 
```

> [!TIP] 
> Different subsets may have different layers which work best. Based on our findings, these are the best layers for classifying the training categories:
> - SupViT: 59
> - DINO: 59
> - MAE: 64
> - ResNet: 59


> [!TIP] 
> Different tasks will likely have different best layers. If trying to classify different attributes (e.g., augmentation use, specific biases, optimization parameters) you should experiment with **all** layers and choose the best ones. 
 
### Generative Splits
ProbeX can be trained on the generative splits of the Model-J dataset to classify LoRA models using CLIP-based zero-shot classification. The training is done using the `train_generative_probex.py` script.

Similar to the discriminative setting, ProbeX trains an individual model per layer. See `sbatch_run_generative_probex.sh` for an example of distributing the training across multiple GPUs.

```bash
python train_generative_probex.py --input_path="~/.cache/huggingface/assets/ProbeX/ModelJ/default/models/SD_200" 
--output_path="ProbeX_outputs/SD_200/results" --subset=SD_200 --start_layer=46 --n_layers=1
```

> [!TIP] 
> Based on our findings, the best layer for classifying the training categories in the generative splits is layer **46**.

#### Downstream Tasks
After training, the learned representations can be evaluated on downstream tasks using `downstream_generative.py`:
```bash
python downstream_generative.py --task=all --input_path="~/.cache/huggingface/assets/ProbeX/ModelJ/models/SD_1k/" 
--checkpoint_path="./checkpoints/best_val_layer-46.safetensors" --subset=SD_1k --layer_idx=46
```
Supported tasks: `occ` (one-class classification), `occ_ledoit`, `knn`, `retrieval`, `all`.

___

## Citation
If you find this useful for your research, please use the following.

```
@InProceedings{Horwitz_2025_CVPR,
    author    = {Horwitz, Eliahu and Cavia, Bar and Kahana, Jonathan and Hoshen, Yedid},
    title     = {Learning on Model Weights using Tree Experts},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {20468-20478}
}
```


## Acknowledgments
- The project makes extensive use of the different Hugging Face libraries (e.g. [Diffusers](https://huggingface.co/docs/diffusers/en/index), [PEFT](https://huggingface.co/docs/peft/en/index), [Transformers](https://huggingface.co/docs/transformers/en/index)).
- The [Model-J dataset](https://huggingface.co/ProbeX) is hosted on Hugging Face.


