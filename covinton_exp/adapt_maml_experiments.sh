#!/bin/bash

# Set the account name
#SBATCH -A bcrm-tgirails
# Set the job name
#SBATCH --job-name=run_maml_experiments
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
#SBATCH --time=03:00:00
# Set the output file
#SBATCH --output=maml_experiments_%j.txt
#SBATCH --dependency=afterok:2007 

# Activate the Conda environment
source /u/nathanj/.bashrc
conda activate meta_learning

# conda info --envs

cd /u/nathanj/meta-learning-streamline-delineation/scripts

python mass_adapt_maml.py
