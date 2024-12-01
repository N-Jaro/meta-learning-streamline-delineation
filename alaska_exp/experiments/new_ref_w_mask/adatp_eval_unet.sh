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
# Set the output file
#SBATCH --output=../logs/new_data_adapt_eval_%j.txt

# Activate the Conda environment
source /u/nathanj/.bashrc
conda activate meta_learning

# Change to the directory containing the batch script
cd /u/nathanj/meta-learning-streamline-delineation/alaska_exp/experiments || exit

# Define paths and parameters
model_path="/u/nathanj/meta-learning-streamline-delineation/alaska_exp/models/new_ref_w_mask_exp/maml_3_500_1_20241107_165940/maml_model.keras" # Unet model path
csv_folder="/u/nathanj/meta-learning-streamline-delineation/alaska_exp/data_gen/within_clusters_clusters/5_kmean_clusters/"
data_dir="/u/nathanj/meta-learning-streamline-delineation/alaska_exp/data_gen_3/huc_code_data_znorm_128/"
model_save_path="/u/nathanj/meta-learning-streamline-delineation/alaska_exp/models/new_ref_w_mask_exp/maml_3_500_1_20241107_165940/"
wandb_project="Alaska_maml_new_ref_w_mask_exp_adapt_eval"
learning_rate=0.000475
inner_steps=1000
channels="0 1 2 4 6 7 8 9 10 11"

# Run the Python script
python /u/nathanj/meta-learning-streamline-delineation/alaska_exp/alaska_clusters_adapt_eval_within_clusters.py \
    --model_path "$model_path" \
    --csv_folder "$csv_folder" \
    --data_dir "$data_dir" \
    --learning_rate "$learning_rate" \
    --model_save_path "$model_save_path" \
    --wandb_project "$wandb_project" \
    --inner_steps "$inner_steps" \
    --channels $channels
