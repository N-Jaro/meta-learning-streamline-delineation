import os
import random
import numpy as np
import rasterio
from rtree import index

# example usage: 
# python generate_patch_locations.py 1000 200 path/to/tiff/file.tif 224 output/directory

def generate_patch_locations(num_samples, tiff_data, patch_size, min_pixels=50):
    rows, cols = tiff_data.shape
    patch_locations = []
    
    while len(patch_locations) < num_samples:
        min_row = random.randint(0, rows - patch_size)
        min_col = random.randint(0, cols - patch_size)
        patch = tiff_data[min_row:min_row + patch_size, min_col:min_col + patch_size]
        
        if np.sum(patch == 1) >= min_pixels:
            patch_locations.append((min_row, min_col))
    
    return patch_locations

def check_overlap(patch_locations, patch_size, idx):
    for min_row, min_col in patch_locations:
        if list(idx.intersection((min_row, min_col, min_row + patch_size, min_col + patch_size))):
            return True
    return False

def write_locations_to_file(locations, output_file):
    with open(output_file, 'w') as f:
        for loc in locations:
            f.write(f"{loc[0]},{loc[1]}\n")

def main(num_train_samples, num_val_samples, tiff_path, patch_size, output_dir):
    with rasterio.open(tiff_path) as src:
        tiff_data = src.read(1)
    
    train_locations = generate_patch_locations(num_train_samples, tiff_data, patch_size)
    
    idx = index.Index()
    for i, (min_row, min_col) in enumerate(train_locations):
        idx.insert(i, (min_row, min_col, min_row + patch_size, min_col + patch_size))
    
    val_locations = []
    while len(val_locations) < num_val_samples:
        potential_loc = generate_patch_locations(1, tiff_data, patch_size)[0]
        if not check_overlap([potential_loc], patch_size, idx):
            val_locations.append(potential_loc)
            idx.insert(len(train_locations) + len(val_locations) - 1, 
                        (potential_loc[0], potential_loc[1], potential_loc[0] + patch_size, potential_loc[1] + patch_size))
    
    tiff_name = os.path.splitext(os.path.basename(tiff_path))[0]
    train_output_file = os.path.join(output_dir, f"{tiff_name}_training_{patch_size}.txt")
    val_output_file = os.path.join(output_dir, f"{tiff_name}_validation_{patch_size}.txt")
    
    write_locations_to_file(train_locations, train_output_file)
    write_locations_to_file(val_locations, val_output_file)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate patch locations for training and validation.")
    parser.add_argument('num_train_samples', type=int, help="Number of training samples")
    parser.add_argument('num_val_samples', type=int, help="Number of validation samples")
    parser.add_argument('tiff_path', type=str, help="Path to the TIF file")
    parser.add_argument('patch_size', type=int, help="Patch size (used for both width and height)")
    parser.add_argument('output_dir', type=str, help="Output directory for the text files")
    
    args = parser.parse_args()
    
    main(args.num_train_samples, args.num_val_samples, args.tiff_path, args.patch_size, args.output_dir)
