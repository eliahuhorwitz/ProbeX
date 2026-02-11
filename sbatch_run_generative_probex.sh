#!/bin/bash
#SBATCH --mem=15gb
#SBATCH -c4
#SBATCH --time=12:00:00
#SBATCH --gres=gg:g0:1
#SBATCH --output=./slurm_logs/SD_200/%A_id%a.txt
#SBATCH --array=0-95:1%96
#SBATCH --killable
#SBATCH --requeue

suffix="${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

echo "Running on layer $SLURM_ARRAY_TASK_ID, suffix=$suffix, SD_200..."
python train_generative_probex.py \
--n_layers=1 \
--input_path="~/.cache/huggingface/assets/ProbeX/ModelJ/models/SD_200/" \
--output_path="RESULTS/SD_200/" \
--subset=SD_200 \
--start_layer=$SLURM_ARRAY_TASK_ID \
--proj_dim=128 \
--n_probes=128 \
--rep_dim=512 \
--suffix=$suffix
