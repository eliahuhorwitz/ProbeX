#!/bin/bash
#SBATCH --mem=15gb
#SBATCH -c4
#SBATCH --time=12:00:00
#SBATCH --gres=gg:g0:1
#SBATCH --output=./slurm_logs/DINO/%A_id%a.txt
#SBATCH --array=0-76:1%76
#SBATCH --killable
#SBATCH --requeue

suffix="${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

echo "Running on layer $SLURM_ARRAY_TASK_ID, suffix=$suffix, DINO..."
python train_discriminative_probex.py \
--n_layers=1 \
--input_path="~/.cache/huggingface/assets/ProbeX/ModelJ/default/models/SupViT/" \
--output_path="RESULTS/SupViT/" \
--is_resnet=False \
--start_layer=$SLURM_ARRAY_TASK_ID \
--proj_dim=128 \
--n_probes=128 \
--rep_dim=128 \
--suffix=$suffix
