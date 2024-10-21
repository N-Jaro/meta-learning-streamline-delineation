#!/bin/bash

# Set the account name
#SBATCH -A bcrm-tgirails
# Set the job name
#SBATCH --job-name=run_patch_extraction
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
#SBATCH --time=03:00:00
# Set the output file
#SBATCH --output=patch_extraction_%j.txt

# Activate the Conda environment
source /u/nathanj/.bashrc
conda activate meta_learning

cd /u/nathanj/meta-learning-streamline-delineation/scripts/Alaska/data_gen

# Set the paths to the data folder, output directory, and the Python script
DATA_FOLDER="/projects/bcrm/nathanj/TIFF_data/Alaska"
OUTPUT_DIR="/u/nathanj/meta-learning-streamline-delineation/scripts/Alaska/data_gen/huc_code_data_128"

# Run the Python script
python generate_patch_locations_tifs.py --data_folder $DATA_FOLDER --patch_size 128 --output_dir $OUTPUT_DIR
