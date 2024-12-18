#!/bin/bash

# Set the account name
#SBATCH -A bcrm-tgirails
# Set the job name
#SBATCH --job-name=alaska_w_254_255_unet
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
#SBATCH --output=logs/alaska_w_254_255_unet_%j.txt

# Activate the Conda environment
source /u/nathanj/.bashrc
conda activate meta_learning

# Change to the directory containing the batch script
cd /u/nathanj/meta-learning-streamline-delineation/alaska_exp/maml_exp/w_254_255/ || exit

python alaska_clusters_adapt_eval_within_clusters.py

# Note params used in this script.    
# parser = argparse.ArgumentParser(description="Automate adaptation and evaluation of MAML model based on HUC codes in CSV files.")
# parser.add_argument('--model_path', type=str, default='/u/nathanj/meta-learning-streamline-delineation/alaska_exp/maml_exp/w_254_255/model/maml_3_500_1_20241203_120625/maml_model.keras', help='meta-learning model')
# parser.add_argument('--model_save_path', type=str, default='/u/nathanj/meta-learning-streamline-delineation/alaska_exp/maml_exp/w_254_255/model/maml_3_500_1_20241203_120625/', help='Path to save adapted models')
# parser.add_argument('--eval_save_paths', type=str, default='/u/nathanj/meta-learning-streamline-delineation/alaska_exp/maml_exp/w_254_255/model/maml_3_500_1_20241203_120625/eval_adapt_results/', help='Path to save evaluations')
# parser.add_argument('--data_dir', type=str, default='/u/nathanj/meta-learning-streamline-delineation/alaska_exp/data/data_w_254_255/huc_code_data_znorm_128/', help='Directory containing the .npy files')
# parser.add_argument('--wandb_project', type=str, default="alaska_w_254_255_unet", help='WandB project name for logging')

# parser.add_argument('--csv_folder', type=str, default='/u/nathanj/meta-learning-streamline-delineation/alaska_exp/data_gen/within_clusters_clusters/5_kmean_clusters/', help='Folder containing CSV files to process')
# parser.add_argument('--inner_steps', type=int, default=250, help='Number of adaptation steps')
# parser.add_argument('--learning_rate', type=float, default=0.0035, help='Learning rate for adaptation')
# parser.add_argument('--normalization_type', type=str, default='-1', choices=['0', '-1', 'none'], help="Normalization type: '0', '-1', or 'none'")
# parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
# parser.add_argument('--channels', type=int, nargs='+', default=[0, 1, 2, 4, 6, 7, 8, 9, 10], help='List of channels to use in the dataset (e.g., 0 1 2 4 6 7 8 9 10)')
