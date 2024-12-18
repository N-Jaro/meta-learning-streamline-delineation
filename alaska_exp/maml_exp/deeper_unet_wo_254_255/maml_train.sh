#!/bin/bash

# Set the account name
#SBATCH -A bcrm-tgirails
# Set the job name
#SBATCH --job-name=run_all_exps
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
#SBATCH --output=logs/maml_wo_254_255_deeperUnet_%j.txt

# Activate the Conda environment
source /u/nathanj/.bashrc
conda activate meta_learning

# Change to the directory containing the batch script
cd /u/nathanj/meta-learning-streamline-delineation/alaska_exp/maml_exp/wo_254_255/ || exit
# Set variables
model="deeperUnet"
save_path="/u/nathanj/meta-learning-streamline-delineation/alaska_exp/maml_exp/deeper_unet_drpoout_wo_254_255/model"
wandb_project_name="alaska_wo_254_255_deeperunet"

# Run the Python script using the full path
python alaska_maml_trianing_within_clusters.py \
    --model "$model" \
    --save_path "$save_path" \
    --wandb_project_name "$wandb_project_name" 