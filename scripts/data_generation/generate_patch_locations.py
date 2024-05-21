import argparse
import os
import random
import rasterio
import numpy as np
from rtree import index

def read_tif(file_path):
    with rasterio.open(file_path) as src:
        return src.read(1)

def is_valid_patch(array, min_row, min_col, patch_size, min_ones=50):
    patch = array[min_row:min_row + patch_size, min_col:min_col + patch_size]
    return np.sum(patch == 1) >= min_ones

def generate_patches(array, num_samples, patch_size):
    rows, cols = array.shape
    samples = []
    
    while len(samples) < num_samples:
        min_row = random.randint(0, rows - patch_size)
        min_col = random.randint(0, cols - patch_size)
        
        if is_valid_patch(array, min_row, min_col, patch_size):
            samples.append((min_row, min_col))
    
    return samples

def save_samples(samples, output_file):
    with open(output_file, 'w') as f:
        for min_row, min_col in samples:
            f.write(f"{min_row},{min_col}\n")

def create_rtree(samples, patch_size):
    idx = index.Index()
    for i, (min_row, min_col) in enumerate(samples):
        idx.insert(i, (min_col, min_row, min_col + patch_size, min_row + patch_size))
    return idx

def generate_validation_patches(array, num_samples, patch_size, train_rtree):
    rows, cols = array.shape
    samples = []
    
    while len(samples) < num_samples:
        min_row = random.randint(0, rows - patch_size)
        min_col = random.randint(0, cols - patch_size)
        
        if is_valid_patch(array, min_row, min_col, patch_size):
            rect = (min_col, min_row, min_col + patch_size, min_row + patch_size)
            if not list(train_rtree.intersection(rect)):
                samples.append((min_row, min_col))
    
    return samples

def main(num_train, num_val, tif_path, patch_size, output_path):
    array = read_tif(tif_path)
    
    train_samples = generate_patches(array, num_train, patch_size)
    if len(train_samples) < num_train:
        print("Not enough training samples available.")
        return
    
    train_rtree = create_rtree(train_samples, patch_size)
    val_samples = generate_validation_patches(array, num_val, patch_size, train_rtree)
    if len(val_samples) < num_val:
        print("Not enough validation samples available.")
        return
    
    base_name = os.path.basename(tif_path).replace('.tif', '')
    train_output_file = os.path.join(output_path, f"{base_name}_train_{patch_size}.txt")
    val_output_file = os.path.join(output_path, f"{base_name}_val_{patch_size}.txt")
    
    save_samples(train_samples, train_output_file)
    save_samples(val_samples, val_output_file)
    print(f"Training and validation sample locations saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate patch locations for training and validation data.")
    parser.add_argument('num_train', type=int, help="Number of training samples")
    parser.add_argument('num_val', type=int, help="Number of validation samples")
    parser.add_argument('tif_path', type=str, help="Path to the input TIFF file")
    parser.add_argument('patch_size', type=int, help="Size of the patch (used for both width and height)")
    parser.add_argument('output_path', type=str, help="Path to the output directory")
    
    args = parser.parse_args()
    main(args.num_train, args.num_val, args.tif_path, args.patch_size, args.output_path)
