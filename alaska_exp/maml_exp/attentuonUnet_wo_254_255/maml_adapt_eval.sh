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
#SBATCH --output=logs/eval_wo_254_255_deeperUnet_%j.txt

# Activate the Conda environment
source /u/nathanj/.bashrc
conda activate meta_learning

# Change to the directory containing the batch script
cd /u/nathanj/meta-learning-streamline-delineation/alaska_exp/maml_exp/wo_254_255/ || exit

# Define paths and parameters
model_path="/u/nathanj/meta-learning-streamline-delineation/alaska_exp/maml_exp/attentuonUnet_wo_254_255/model/maml_3_500_1_20241206_113305/maml_model.keras" # Unet model path
csv_folder="/u/nathanj/meta-learning-streamline-delineation/alaska_exp/data_gen/within_clusters_clusters/5_kmean_clusters/"
eval_save_paths="/u/nathanj/meta-learning-streamline-delineation/alaska_exp/maml_exp/attentuonUnet_wo_254_255/model/maml_3_500_1_20241206_113305/eval_adapt_results_4/"
model_save_path="/u/nathanj/meta-learning-streamline-delineation/alaska_exp/maml_exp/attentuonUnet_wo_254_255/model/maml_3_500_1_20241206_113305/eval_adapt_results_4/"
wandb_project="alaska_wo_254_255_attentionUnet_4"
learning_rate=0.000275
inner_steps=1000

# Run the Python script
python alaska_clusters_adapt_eval_within_clusters.py \
    --model_path "$model_path" \
    --csv_folder "$csv_folder" \
    --eval_save_paths "$eval_save_paths" \
    --model_save_path "$model_save_path" \
    --wandb_project "$wandb_project" \
    --learning_rate "$learning_rate" \
    --inner_steps "$inner_steps"
