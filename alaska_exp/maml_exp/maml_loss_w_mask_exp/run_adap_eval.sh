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
#SBATCH --output=logs/adapt_eval_output_%j.txt

# Activate the Conda environment
source /u/nathanj/.bashrc
conda activate meta_learning

# Run the Python script
python /u/nathanj/meta-learning-streamline-delineation/alaska_exp/experiment_w_mask/alaska_clusters_adapt_eval_within_clusters.py
