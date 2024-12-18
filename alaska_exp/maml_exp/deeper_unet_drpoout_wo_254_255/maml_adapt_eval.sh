#!/bin/bash

# Set the account name
#SBATCH -A bcrm-tgirails
# Set the job name
#SBATCH --job-name=deeperUnet_dropout_wo_254_255
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
#SBATCH --output=logs/eval_wo_254_255_deeperUnet_dropout_%j.txt

# Activate the Conda environment
source /u/nathanj/.bashrc
conda activate meta_learning

# Change to the directory containing the batch script
cd /u/nathanj/meta-learning-streamline-delineation/alaska_exp/maml_exp/wo_254_255/ || exit

# Define paths and parameters
model_path="/u/nathanj/meta-learning-streamline-delineation/alaska_exp/maml_exp/deeper_unet_drpoout_wo_254_255/model/maml_3_500_1_20241204_104123/maml_model.keras" # Unet model path
csv_folder="/u/nathanj/meta-learning-streamline-delineation/alaska_exp/data_gen/within_clusters_clusters/5_kmean_clusters/"
eval_save_paths="/u/nathanj/meta-learning-streamline-delineation/alaska_exp/maml_exp/deeper_unet_drpoout_wo_254_255/model/maml_3_500_1_20241204_104123/eval_adapt_results/"
model_save_path="/u/nathanj/meta-learning-streamline-delineation/alaska_exp/maml_exp/deeper_unet_drpoout_wo_254_255/model/maml_3_500_1_20241204_104123/"
wandb_project="alaska_wo_254_255_deeperunet_dropout"
learning_rate=0.000035

# Run the Python script
python alaska_clusters_adapt_eval_within_clusters.py \
    --model_path "$model_path" \
    --csv_folder "$csv_folder" \
    --eval_save_paths "$eval_save_paths" \
    --model_save_path "$model_save_path" \
    --learning_rate "$learning_rate" \
    --wandb_project "$wandb_project" 
