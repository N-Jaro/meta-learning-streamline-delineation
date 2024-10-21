import os
import sys
import argparse
import rasterio
import numpy as np
from rtree import index

def preprocess_tif(tif_data):
    """Replace values greater than 200 with 1 and values less than 0 with 0."""
    tif_data[tif_data != 1] = 0
    return tif_data

def create_grid_patches(tif_data, patch_size):
    """Create grid patches and find valid patches that contain at least one pixel with value 1."""
    valid_patches = []
    for row in range(0, tif_data.shape[0] - patch_size + 1, patch_size):
        for col in range(0, tif_data.shape[1] - patch_size + 1, patch_size):
            patch = tif_data[row:row + patch_size, col:col + patch_size]
            if np.any(patch == 1):
                valid_patches.append((row, col))
    return valid_patches

def split_patches(valid_patches):
    """Split the valid patches into training (70%) and validation (30%) sets."""
    np.random.shuffle(valid_patches)
    split_idx = int(0.7 * len(valid_patches))
    training_patches = valid_patches[:split_idx]
    validation_patches = valid_patches[split_idx:]
    return training_patches, validation_patches

def generate_locations(args):
    """Main function to generate patch locations and save to text files."""
    with rasterio.open(args.tif_path) as src:
        tif_data = src.read(1)

    tif_data = preprocess_tif(tif_data)
    valid_patches = create_grid_patches(tif_data, args.patch_size)

    if not valid_patches:
        print("No valid patches found.")
        return

    training_patches, validation_patches = split_patches(valid_patches)

    training_file = os.path.join(args.output_dir, f"{os.path.splitext(os.path.basename(args.tif_path))[0]}_training_{args.patch_size}.txt")
    validation_file = os.path.join(args.output_dir, f"{os.path.splitext(os.path.basename(args.tif_path))[0]}_validation_{args.patch_size}.txt")

    with open(training_file, 'w') as f:
        for min_row, min_col in training_patches:
            f.write(f"{min_row},{min_col}\n")

    with open(validation_file, 'w') as f:
        for min_row, min_col in validation_patches:
            f.write(f"{min_row},{min_col}\n")

    print(f"Training samples: {len(training_patches)}")
    print(f"Validation samples: {len(validation_patches)}")
    print(f"Training samples saved to {training_file}")
    print(f"Validation samples saved to {validation_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate patch locations for training and validation datasets.")
    parser.add_argument("--tif_path", type=str, required=True, help="Path to the TIFF file")
    parser.add_argument("--patch_size", type=int, default=224, help="Size of the patch (default: 224)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for the text files")
    
    args = parser.parse_args()
    generate_locations(args)