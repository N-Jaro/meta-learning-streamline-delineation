import os
import sys
import argparse
import rasterio
import numpy as np
from rtree import index

def create_grid_patches(tif_data, patch_size):
    """
    Create patches in a grid pattern over the image and filter out invalid patches.
    A valid patch contains only pixels of 0 or 1 and no NaN values.
    
    Parameters:
    tif_data (numpy array): The array representation of the TIFF file.
    patch_size (int): The size of the patches to be created.

    Returns:
    List of tuples: A list of valid patch coordinates (min_row, min_col).
    """
    valid_patches = []
    n_rows, n_cols = tif_data.shape
    for min_row in range(0, n_rows - patch_size + 1, patch_size):
        for min_col in range(0, n_cols - patch_size + 1, patch_size):
            patch = tif_data[min_row:min_row + patch_size, min_col:min_col + patch_size]
            if np.all((patch == 0) | (patch == 1)) and not np.any(np.isnan(patch)):
                valid_patches.append((min_row, min_col))
    return valid_patches

def generate_locations(args):
    """
    Generate patch locations for training and validation datasets from a TIFF file.
    The patches are created in a grid pattern and split into training (70%) and validation (30%) sets.

    Parameters:
    args (argparse.Namespace): The command-line arguments parsed by argparse.
    """
    # Read the TIFF file
    with rasterio.open(args.tif_path) as src:
        tif_data = src.read(1)

    # Create valid patches in a grid pattern
    patch_size = args.patch_size
    valid_patches = create_grid_patches(tif_data, patch_size)
    num_samples = len(valid_patches)
    
    # Check if there are enough valid patches
    if num_samples == 0:
        print("No valid patches found.")
        return

    # Shuffle and split patches into training (70%) and validation (30%) sets
    np.random.shuffle(valid_patches)
    num_train_samples = int(0.7 * num_samples)
    num_val_samples = num_samples - num_train_samples
    training_patches = valid_patches[:num_train_samples]
    validation_patches = valid_patches[num_train_samples:]

    # Output file paths
    training_file = os.path.join(args.output_dir, f"{os.path.splitext(os.path.basename(args.tif_path))[0]}_training_{patch_size}.txt")
    validation_file = os.path.join(args.output_dir, f"{os.path.splitext(os.path.basename(args.tif_path))[0]}_validation_{patch_size}.txt")

    # Write training patch coordinates to file
    with open(training_file, 'w') as f:
        for min_row, min_col in training_patches:
            f.write(f"{min_row},{min_col}\n")

    # Write validation patch coordinates to file
    with open(validation_file, 'w') as f:
        for min_row, min_col in validation_patches:
            f.write(f"{min_row},{min_col}\n")

    # Print the number of training and validation patches
    print(f"Training samples: {num_train_samples}")
    print(f"Validation samples: {num_val_samples}")
    print(f"Training samples saved to {training_file}")
    print(f"Validation samples saved to {validation_file}")

if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Generate patch locations for training and validation datasets.")
    parser.add_argument("--tif_path", type=str, required=True, help="Path to the TIFF file")
    parser.add_argument("--patch_size", type=int, default=224, help="Size of the patch (default: 224)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for the text files")
    
    # Parse arguments and generate locations
    args = parser.parse_args()
    generate_locations(args)
