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
#SBATCH --output=logs/Alaska_maml_new_ref_w_mask_exp_%j.txt

# Activate the Conda environment
source /u/nathanj/.bashrc
conda activate meta_learning

# Change to the directory containing the batch script
cd /u/nathanj/meta-learning-streamline-delineation/alaska_exp/experiments/new_ref_w_mask || exit

# Set variables
data_dir="/u/nathanj/meta-learning-streamline-delineation/alaska_exp/data_gen_3/huc_code_data_znorm_128"
training_csv="/u/nathanj/meta-learning-streamline-delineation/alaska_exp/data_gen/within_clusters_clusters/5_kmean_clusters/huc_code_kmean_5_train.csv"
model="unet"
inner_lr=0.00078
meta_lr=0.000359
save_path="/u/nathanj/meta-learning-streamline-delineation/alaska_exp/models/new_ref_w_mask_exp/"
wandb_project_name="Alaska_maml_new_ref_w_mask_exp"
channels="0 1 2 4 6 7 8 9 10 11"
decay_steps=1500

# Run the Python script using the full path
python /u/nathanj/meta-learning-streamline-delineation/alaska_exp/alaska_maml_trianing_within_clusters.py \
    --data_dir "$data_dir" \
    --training_csv "$training_csv" \
    --model "$model" \
    --inner_lr "$inner_lr" \
    --meta_lr "$meta_lr" \
    --save_path "$save_path" \
    --wandb_project_name "$wandb_project_name" \
    --decay_steps "$decay_steps" \
    --channels $channels
