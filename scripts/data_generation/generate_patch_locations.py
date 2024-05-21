import argparse
import os
import numpy as np
import rasterio
from rtree import index
import random

def check_patch(array, row, col, patch_size):
    patch = array[row:row + patch_size, col:col + patch_size]
    return np.sum(patch == 1) >= 50

def generate_samples(array, num_samples, patch_size, existing_samples=None):
    rows, cols = array.shape
    samples = []
    idx = index.Index()
    
    if existing_samples:
        for i, (row, col) in enumerate(existing_samples):
            idx.insert(i, (row, col, row + patch_size, col + patch_size))
    
    attempts = 0
    max_attempts = num_samples * 100
    while len(samples) < num_samples and attempts < max_attempts:
        row = random.randint(0, rows - patch_size)
        col = random.randint(0, cols - patch_size)
        if check_patch(array, row, col, patch_size):
            if existing_samples:
                intersect = list(idx.intersection((row, col, row + patch_size, col + patch_size)))
                if intersect:
                    attempts += 1
                    continue
            samples.append((row, col))
            idx.insert(len(samples) - 1, (row, col, row + patch_size, col + patch_size))
        attempts += 1
    
    if len(samples) < num_samples:
        print(f"Not enough samples could be generated. Only {len(samples)} samples generated out of {num_samples}.")
    
    return samples

def save_samples(samples, output_file):
    with open(output_file, 'w') as f:
        for row, col in samples:
            f.write(f"{row},{col}\n")

def main(num_train_samples, num_val_samples, tif_path, patch_size, output_path):
    with rasterio.open(tif_path) as src:
        array = src.read(1)
    
    train_samples = generate_samples(array, num_train_samples, patch_size)
    val_samples = generate_samples(array, num_val_samples, patch_size, train_samples)
    
    if len(train_samples) == num_train_samples and len(val_samples) == num_val_samples:
        tif_name = os.path.splitext(os.path.basename(tif_path))[0]
        train_output_file = os.path.join(output_path, f"{tif_name}_training_{patch_size}.txt")
        val_output_file = os.path.join(output_path, f"{tif_name}_validation_{patch_size}.txt")
        
        save_samples(train_samples, train_output_file)
        save_samples(val_samples, val_output_file)
        
        print("Samples generated successfully.")
    else:
        print("Failed to generate the required number of samples.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate training and validation samples for patches.")
    parser.add_argument('num_train_samples', type=int, help="Number of training samples.")
    parser.add_argument('num_val_samples', type=int, help="Number of validation samples.")
    parser.add_argument('tif_path', type=str, help="Path to the input TIFF file.")
    parser.add_argument('patch_size', type=int, help="Size of the patch (W and H).")
    parser.add_argument('output_path', type=str, help="Path to the output directory for the text files.")
    
    args = parser.parse_args()
    
    main(args.num_train_samples, args.num_val_samples, args.tif_path, args.patch_size, args.output_path)
