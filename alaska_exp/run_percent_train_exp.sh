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
#SBATCH --output=MAML_percent_train_exp_%j.txt

# Activate the Conda environment
source /u/nathanj/.bashrc
conda activate meta_learning

# Array of CSV folder paths, paired with corresponding model_paths
training_csv=(  # Change the variable from csv_folders to training_csv
    "/u/nathanj/meta-learning-streamline-delineation/alaska_exp/data_gen/within_clusters_clusters/1_percent_train/huc_code_kmean_5_train.csv"
    "/u/nathanj/meta-learning-streamline-delineation/alaska_exp/data_gen/within_clusters_clusters/5_percent_train/huc_code_kmean_5_train.csv"
    "/u/nathanj/meta-learning-streamline-delineation/alaska_exp/data_gen/within_clusters_clusters/10_percent_train/huc_code_kmean_5_train.csv"
    "/u/nathanj/meta-learning-streamline-delineation/alaska_exp/data_gen/within_clusters_clusters/25_percent_train/huc_code_kmean_5_train.csv"
)

# Array of corresponding save paths
save_paths=(
    "/u/nathanj/meta-learning-streamline-delineation/alaska_exp/models/1_percent_train/"
    "/u/nathanj/meta-learning-streamline-delineation/alaska_exp/models/5_percent_train/"
    "/u/nathanj/meta-learning-streamline-delineation/alaska_exp/models/10_percent_train/"
    "/u/nathanj/meta-learning-streamline-delineation/alaska_exp/models/25_percent_train/"
)

# Array of num_episodes corresponding to each experiment
num_episodes=(
    5
    25
    50
    100
)

# Array of num_episodes corresponding to each experiment
decay_steps=(
    500
    1000
    1500
    2000
)

# Ensure arrays are of the same length
if [ "${#training_csv[@]}" -ne "${#save_paths[@]}" ] || [ "${#training_csv[@]}" -ne "${#num_episodes[@]}" ]; then  # Change the array name to training_csv
    echo "Error: The number of CSV files, save paths, and num_episodes must be the same."
    exit 1
fi

# Function to run the process and wait for available slot to continue
function run_process {
    csv_file="$1"   # Change the variable name to csv_file for clarity
    save_path="$2"
    num_episode="$3"
    decay_step="$4"

    echo "Running adaptation and evaluation"
    echo "Using training CSV: $csv_file"
    echo "Saving results to: $save_path"
    echo "Using num_episodes: $num_episode"
    echo "Using decay_step: $decay_step"

    python alaska_maml_trianing_within_clusters.py --training_csv "$csv_file" --decay_steps "$decay_step" --save_path "$save_path" --num_episodes "$num_episode" &
}

# Limit the number of concurrent jobs to 2
MAX_JOBS=2
job_count=0

# Loop over training_csv, save_paths, and num_episodes, and ensure 2 jobs run concurrently
for i in "${!training_csv[@]}"; do  # Change the array reference to training_csv
    csv_file="${training_csv[$i]}"  # Change the variable name to csv_file
    save_path="${save_paths[$i]}"
    num_episode="${num_episodes[$i]}"
    decay_step="${decay_steps[$i]}"

    run_process "$csv_file" "$save_path" "$num_episode" "$decay_steps"
    ((job_count++))

    # If the number of running jobs reaches the limit, wait for one to finish
    if [ "$job_count" -ge "$MAX_JOBS" ]; then
        wait -n  # Wait for any job to finish
        ((job_count--))  # Decrease the job count when one finishes
    fi
done

# Wait for remaining background jobs to finish
wait

# Optional: If you want to print out the conda environment info (commented out)
# conda info --envs
