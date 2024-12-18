#!/bin/bash

# Set the account name
#SBATCH -A bcrm-tgirails
# Set the job name
#SBATCH --job-name=scratch_deeper_wo_254_255
# Set the partition
#SBATCH --partition=gpu
# Set the number of nodes
#SBATCH --nodes=1
# Set the number of tasks per node
#SBATCH --ntasks=1
# Set the number of CPUs per task
#SBATCH --cpus-per-task=16
# Set the number of GPUs
#SBATCH --gpus=2
# Set the amount of memory

#SBATCH --mem=100GB
# Set the time limit (hh:mm:ss)
#SBATCH --time=10:00:00
# Set the output file
#SBATCH --output=logs/scratch_deeper_wo_254_255_%j.txt

# Activate the Conda environment
source /u/nathanj/.bashrc
conda activate meta_learning

# Change to the directory containing the batch script
cd /u/nathanj/meta-learning-streamline-delineation/alaska_exp/train_scratch_exp/wo_254_255/ || exit

model_save_dir='/u/nathanj/meta-learning-streamline-delineation/alaska_exp/train_scratch_exp/attentionUnet_wo_254_255/model/'
metrics_save_dir='/u/nathanj/meta-learning-streamline-delineation/alaska_exp/train_scratch_exp/attentionUnet_wo_254_255/model/eval_results'
model="attentionUnet"
wandb_project='alaska_attentionUnet_wo_254_255_scratch'


# Run the Python script
python alaska_train_scratch.py \
    --model "$model" \
    --model_save_dir "$model_save_dir" \
    --metrics_save_dir "$metrics_save_dir" \
    --wandb_project "$wandb_project"