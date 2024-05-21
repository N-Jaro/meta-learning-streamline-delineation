import os
import sys
import argparse
import rasterio
import numpy as np
from rtree import index

def create_patch_locations(num_samples, patch_size, tif_data, tries_limit, num_stream_pixels):
    patches = []
    num_tries = 0
    while len(patches) < num_samples and num_tries < tries_limit:
        min_row = np.random.randint(0, tif_data.shape[0] - patch_size)
        min_col = np.random.randint(0, tif_data.shape[1] - patch_size)
        patch = tif_data[min_row:min_row + patch_size, min_col:min_col + patch_size]
        if np.sum(patch) >= num_stream_pixels:
            patches.append((min_row, min_col))
        num_tries += 1
    return patches

def generate_locations(args):
    with rasterio.open(args.tif_path) as src:
        tif_data = src.read(1)

    tries_limit = args.num_train_samples * 100
    training_patches = create_patch_locations(args.num_train_samples, args.patch_size, tif_data, tries_limit, args.num_strem_px)

    if len(training_patches) < args.num_train_samples:
        print("Not enough training samples could be generated.")
        return

    idx = index.Index()
    for i, (min_row, min_col) in enumerate(training_patches):
        idx.insert(i, (min_col, min_row, min_col + args.patch_size, min_row + args.patch_size))

    validation_patches = []
    num_tries = 0
    while len(validation_patches) < args.num_val_samples and num_tries < tries_limit:
        min_row = np.random.randint(0, tif_data.shape[0] - args.patch_size)
        min_col = np.random.randint(0, tif_data.shape[1] - args.patch_size)
        patch_bounds = (min_col, min_row, min_col + args.patch_size, min_row + args.patch_size)
        if list(idx.intersection(patch_bounds)):
            num_tries += 1
            continue
        patch = tif_data[min_row:min_row + args.patch_size, min_col:min_col + args.patch_size]
        if np.sum(patch) >= 50:
            validation_patches.append((min_row, min_col))
            idx.insert(len(training_patches) + len(validation_patches), patch_bounds)
        num_tries += 1

    if len(validation_patches) < args.num_val_samples:
        print("Not enough validation samples could be generated.")
        return

    training_file = os.path.join(args.output_dir, f"{os.path.splitext(os.path.basename(args.tif_path))[0]}_training_{args.patch_size}.txt")
    validation_file = os.path.join(args.output_dir, f"{os.path.splitext(os.path.basename(args.tif_path))[0]}_validation_{args.patch_size}.txt")

    with open(training_file, 'w') as f:
        for min_row, min_col in training_patches:
            f.write(f"{min_row},{min_col}\n")

    with open(validation_file, 'w') as f:
        for min_row, min_col in validation_patches:
            f.write(f"{min_row},{min_col}\n")

    print(f"Training samples saved to {training_file}")
    print(f"Validation samples saved to {validation_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate patch locations for training and validation datasets.")
    parser.add_argument("--num_train_samples", type=int, default=100, required=True, help="Number of training samples")
    parser.add_argument("--num_val_samples", type=int, default=50, required=True, help="Number of validation samples")
    parser.add_argument("--num_strem_px", type=int, default=50, required=True, help="Number of validation samples")
    parser.add_argument("--tif_path", type=str, default="/u/nathanj/projects/nathanj/TIFF_data/Alaska/19050302_50/AK_50_Filtered_Reference/190503021001_filtered_reference.tif", required=True, help="Path to the TIFF file")
    parser.add_argument("--patch_size", type=int, default=224, help="Size of the patch (default: 224)")
    parser.add_argument("--output_dir", type=str, default="./", required=True, help="Output directory for the text files")
    
    args = parser.parse_args()
    generate_locations(args)
