import os
import numpy as np
import rasterio
from rasterio.merge import merge

def read_patch_locations(file_path):
    """Read the patch locations from a text file."""
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # First line is the directory path
    data_dir = lines[0].strip()

    # Remaining lines are the patch locations
    patch_locations = [tuple(map(int, line.strip().split(','))) for line in lines[1:]]

    return data_dir, patch_locations

def stack_tif_files(data_dir, huc_code):
    """Stack specified TIFF files to create an 8-channel array."""
    file_names = [
        f"curvature_{huc_code}.tif",
        f"swm1_{huc_code}.tif",
        f"ori_ave_{huc_code}.tif",
        f"dsm_{huc_code}.tiff",
        f"geomorph_{huc_code}.tif",
        f"pos_openness_{huc_code}.tif",
        f"tpi_3_{huc_code}.tif",
        f"twi_{huc_code}.tif"
    ]

    # Read and stack the TIFF files
    channels = []
    for file_name in file_names:
        file_path = os.path.join(data_dir, file_name)
        with rasterio.open(file_path) as src:
            channels.append(src.read(1))

    stacked_array = np.stack(channels, axis=-1)
    return stacked_array

def extract_patches(stacked_array, patch_locations, patch_size):
    """Extract patches from the stacked array using the provided locations."""
    patches = []
    for row, col in patch_locations:
        patch = stacked_array[row:row + patch_size, col:col + patch_size]
        patches.append(patch)
    return patches

def save_patches(patches, output_dir, base_name):
    """Save the extracted patches to disk."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, patch in enumerate(patches):
        patch_file = os.path.join(output_dir, f"{base_name}_patch_{i}.npy")
        np.save(patch_file, patch)

    print(f"Saved {len(patches)} patches to {output_dir}")

def process_patch_files(patch_files, patch_size, output_dir):
    """Process each patch file to extract and save patches."""
    for patch_file in patch_files:
        data_dir, patch_locations = read_patch_locations(patch_file)
        huc_code = os.path.basename(data_dir)  # Extract the HUC code from the directory path

        # Stack the TIFF files to create an 8-channel array
        stacked_array = stack_tif_files(data_dir, huc_code)

        # Extract patches using the patch locations
        patches = extract_patches(stacked_array, patch_locations, patch_size)

        # Save the extracted patches to disk
        base_name = os.path.splitext(os.path.basename(patch_file))[0]
        save_patches(patches, output_dir, base_name)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract and save patches from stacked TIFF files.")
    parser.add_argument("--patch_files", type=str, nargs='+', required=True, help="Paths to the patch location text files")
    parser.add_argument("--patch_size", type=int, default=224, help="Size of the patch (default: 224)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for the patches")

    args = parser.parse_args()

    process_patch_files(args.patch_files, args.patch_size, args.output_dir)
