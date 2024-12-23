#!/bin/bash

#SBATCH -A bcrm-tgirails          # Set the account name
#SBATCH --job-name=scratch_train    # Set the job name
#SBATCH --partition=gpu          # Set the partition
#SBATCH --nodes=1                # Set the number of nodes
#SBATCH --ntasks=1               # Set the number of tasks per node
#SBATCH --cpus-per-task=16       # Set the number of CPUs per task
#SBATCH --gpus=2                 # Set the number of GPUs
#SBATCH --mem=100GB              # Set the amount of memory
#SBATCH --time=20:00:00          # Set the time limit (hh:mm:ss)
#SBATCH --output=logs/scratch_train_output_%j.txt  # Set the output file

source /u/nathanj/.bashrc
conda activate meta_learning

cd /u/nathanj/meta-learning-streamline-delineation/alaska_exp/train_scratch_exp/wo_254_255/ || exit

for num_samples in 50 -1; do
  for i in $(seq 1 5); do
    # Set the WandB project name based on num_samples
    if [ ${num_samples} -eq -1 ]; then
      wandb_project="alaska_wo_254_255_scratch_all_samples"
    else
      wandb_project="alaska_wo_254_255_scratch_${num_samples}_samples"
    fi

    echo "Running scratch_train.py with num_samples=${num_samples}, run ${i}, wandb_project=${wandb_project}..."
    python scratch_train.py --num_sample_per_huc_code ${num_samples} --wandb_project ${wandb_project}
  done
done