import os
import argparse
import random
import numpy as np
import rasterio
from rtree import index

def check_valid_patch(array, row, col, patch_size):
    patch = array[row:row+patch_size, col:col+patch_size]
    return np.sum(patch == 1) >= 50

def generate_samples(num_samples, array, patch_size, existing_samples=set()):
    samples = []
    max_row, max_col = array.shape[0] - patch_size, array.shape[1] - patch_size
    attempts = 0

    while len(samples) < num_samples and attempts < 10000:
        min_row = random.randint(0, max_row)
        min_col = random.randint(0, max_col)
        if check_valid_patch(array, min_row, min_col, patch_size):
            if not any(min_row <= er + patch_size and min_row + patch_size >= er and
                       min_col <= ec + patch_size and min_col + patch_size >= ec
                       for (er, ec) in existing_samples):
                samples.append((min_row, min_col))
                existing_samples.add((min_row, min_col))
        attempts += 1

    return samples

def save_samples(samples, output_file):
    with open(output_file, 'w') as f:
        for row, col in samples:
            f.write(f"{row},{col}\n")

def main(num_train_samples, num_val_samples, tif_path, patch_size, output_dir):
    with rasterio.open(tif_path) as src:
        array = src.read(1)

    train_samples = generate_samples(num_train_samples, array, patch_size)
    if len(train_samples) < num_train_samples:
        print("Not enough valid training samples found.")
        return

    existing_samples = set(train_samples)
    val_samples = generate_samples(num_val_samples, array, patch_size, existing_samples)
    if len(val_samples) < num_val_samples:
        print("Not enough valid validation samples found.")
        return

    tif_name = os.path.splitext(os.path.basename(tif_path))[0]
    train_output_file = os.path.join(output_dir, f"{tif_name}_training_{patch_size}.txt")
    val_output_file = os.path.join(output_dir, f"{tif_name}_validation_{patch_size}.txt")

    save_samples(train_samples, train_output_file)
    save_samples(val_samples, val_output_file)

    print(f"Training samples saved to {train_output_file}")
    print(f"Validation samples saved to {val_output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate patch locations for training and validation.')
    parser.add_argument('num_train_samples', type=int, help='Number of training samples')
    parser.add_argument('num_val_samples', type=int, help='Number of validation samples')
    parser.add_argument('tif_path', type=str, help='Path to the TIFF file')
    parser.add_argument('patch_size', type=int, help='Patch size (same for width and height)')
    parser.add_argument('output_dir', type=str, help='Output directory for the sample text files')

    args = parser.parse_args()
    main(args.num_train_samples, args.num_val_samples, args.tif_path, args.patch_size, args.output_dir)
