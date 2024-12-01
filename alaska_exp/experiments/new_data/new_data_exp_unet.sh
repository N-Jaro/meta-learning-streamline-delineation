#!/bin/bash

# Set the account name
#SBATCH -A bcrm-tgirails
# Set the job name
#SBATCH --job-name=run_all_exps
# Set the partition
#SBATCH --partition=cpu
# Set the number of nodes
#SBATCH --nodes=1
# Set the number of tasks per node
#SBATCH --ntasks=1
# Set the number of CPUs per task
#SBATCH --cpus-per-task=16
# Set the amount of memory
#SBATCH --mem=100GB
# Set the time limit (hh:mm:ss)
#SBATCH --time=15:00:00
# Set the output file in the ../logs folder
#SBATCH --output=../logs/Alaska_maml_new_data_exp_%j.txt

# Experiment Details:
# This experiment uses new data located in data_gen_2.
# The new data has 11 channels, but we are skipping two channels, specifically ORI and Geomorphons.
# As a result, we are using the following channels: [0,1,2,4,6,7,8,9,10].
# To accommodate this, the model structure was modified to support 9 input channels instead of the original structure.
# Updated model initialization in the training script: setup_model(config['model_type'], input_shape=(128, 128, 9), num_classes=1)

# Activate the Conda environment
source /u/nathanj/.bashrc
conda activate meta_learning

# Change to the directory containing the batch script
cd /u/nathanj/meta-learning-streamline-delineation/alaska_exp/experiments || exit

# Set variables
data_dir="/u/nathanj/meta-learning-streamline-delineation/alaska_exp/data_gen_2/huc_code_data_znorm_128"
training_csv="/u/nathanj/meta-learning-streamline-delineation/alaska_exp/data_gen/within_clusters_clusters/5_kmean_clusters/huc_code_kmean_5_train.csv"
model="unet"
# inner_lr=0.0180
# meta_lr=0.0089
inner_lr=0.00285
meta_lr=0.000475
save_path="/u/nathanj/meta-learning-streamline-delineation/alaska_exp/models/new_data_exp/"
wandb_project_name="Alaska_maml_new_data_exp"

# Run the Python script using the full path
python /u/nathanj/meta-learning-streamline-delineation/alaska_exp/alaska_maml_trianing_within_clusters.py \
    --data_dir "$data_dir" \
    --training_csv "$training_csv" \
    --model "$model" \
    --inner_lr "$inner_lr" \
    --meta_lr "$meta_lr" \
    --save_path "$save_path" \
    --wandb_project_name "$wandb_project_name"
