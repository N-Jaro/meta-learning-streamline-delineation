#!/bin/bash

#SBATCH -A bcrm-tgirails          # Set the account name
#SBATCH --job-name=maml_samp_eps_train    # Set the job name
#SBATCH --partition=cpu          # Set the partition
#SBATCH --nodes=1                # Set the number of nodes
#SBATCH --ntasks=1               # Set the number of tasks per node
#SBATCH --cpus-per-task=16       # Set the number of CPUs per task
#SBATCH --mem=100GB              # Set the amount of memory
#SBATCH --time=48:00:00          # Set the time limit (hh:mm:ss)
#SBATCH --output=../logs/maml_samp_eps_output_%j.txt  # Set the output file

source /u/nathanj/.bashrc
conda activate meta_learning

cd /u/nathanj/meta-learning-streamline-delineation/alaska_exp/maml_exp/organized_exp/ || exit

for num_episodes in 29; do # 5 10 15 25 29 Experiments
  for num_samples in 50; do # 5 10 15 25 50 Experiments
    wandb_project="alaska_wo_254_255_maml_${num_samples}_samples_${num_episodes}_episodes"

    echo "Running maml_training.py with num_samples_per_location=${num_samples}, num_episodes=${num_episodes}, wandb_project_name=${wandb_project} ..."
    python maml_training.py --num_samples_per_location ${num_samples} --num_episodes ${num_episodes} --wandb_project_name ${wandb_project}
  done
done