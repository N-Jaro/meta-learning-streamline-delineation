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
#SBATCH --time=15:00:00
# Set the output file
#SBATCH --output=MAML_adapt_eval_train_samples_%j.txt

# Activate the Conda environment
source /u/nathanj/.bashrc
conda activate meta_learning

# Array of model file paths
model_paths=(
    "/u/nathanj/meta-learning-streamline-delineation/alaska_exp/models/Alaska_within_clusters_simpleattention_15_samples/maml_3_500_1_20241023_094331/maml_model.keras" #SimpleAttentionUnet model path
)

model_save_path="/u/nathanj/meta-learning-streamline-delineation/alaska_exp/models/Alaska_within_clusters_simpleattention_15_samples/maml_3_500_1_20241023_094331/"

wandb_project="Alaska_within_clusters_SimepleAttentionUnet"

learning_rate=0.0000359

inner_steps=1000

# Loop over model file paths and run the Python script with each model
for model_path in "${model_paths[@]}"
do
    echo "Running adaptation and evaluation for model: $model_path"
    python alaska_clusters_adapt_eval_within_clusters.py --model_path "$model_path" --learning_rate "$learning_rate" --model_save_path "$model_save_path" --wandb_project "$wandb_project" --inner_steps "$inner_steps"
done

# Optional: If you want to print out the conda environment info (commented out)
# conda info --envs