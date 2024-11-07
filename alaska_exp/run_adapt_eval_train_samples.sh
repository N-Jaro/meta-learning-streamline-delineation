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
    #"/u/nathanj/meta-learning-streamline-delineation/alaska_exp/models/Alaska_within_clusters_samples_per_ep/maml_3_500_1_20241021_132100/maml_model.keras" #5
    #"/u/nathanj/meta-learning-streamline-delineation/alaska_exp/models/Alaska_within_clusters_samples_per_ep/maml_3_500_1_20241021_163941/maml_model.keras" #10
    # "/u/nathanj/meta-learning-streamline-delineation/alaska_exp/models/Alaska_within_clusters_samples_per_ep/maml_3_500_1_20241021_133949/maml_model.keras" #15
    #"/u/nathanj/meta-learning-streamline-delineation/alaska_exp/models/Alaska_within_clusters_samples_per_ep/maml_3_500_1_20241021_141223/maml_model.keras" #20
    #"/u/nathanj/meta-learning-streamline-delineation/alaska_exp/models/Alaska_within_clusters_samples_per_ep/maml_3_500_1_20241021_153851/maml_model.keras" #25
    # "/u/nathanj/meta-learning-streamline-delineation/alaska_exp/models/Alaska_within_clusters_samples_per_ep/maml_3_500_1_20241021_172036/maml_model.keras" #50_1
    # "/u/nathanj/meta-learning-streamline-delineation/alaska_exp/models/Alaska_within_clusters_samples_per_ep/maml_3_500_1_20241021_181716/maml_model.keras" #50_2
    # Add more paths as needed
)

# Loop over model file paths and run the Python script with each model
for model_path in "${model_paths[@]}"
do
    echo "Running adaptation and evaluation for model: $model_path"
    python alaska_clusters_adapt_eval_within_clusters.py --model_path "$model_path"
done

# Optional: If you want to print out the conda environment info (commented out)
# conda info --envs