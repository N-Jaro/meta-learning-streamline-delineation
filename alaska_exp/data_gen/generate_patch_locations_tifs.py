import os
import argparse
import numpy as np
import rasterio

def preprocess_label_tif(tif_data):
    """Replace values greater than 200 with 1 and values less than 0 with 0."""
    tif_data[tif_data != 1] = 0
    return tif_data

# def normalize_tif(tif_data):
#     """Normalize the TIFF data to a range of 0-255."""
#     min_val = np.min(tif_data)
#     max_val = np.max(tif_data)
#     normalized_data = ((tif_data - min_val) / (max_val - min_val)) * 255
#     return normalized_data.astype(np.uint8)

def normalize_tif(tif_data):
    """Normalize the TIFF data using mean and standard deviation, then scale to 0-255 range."""
    tif_data[tif_data == -9999.0] = 0
    mean_val = np.mean(tif_data)
    std_val = np.std(tif_data)

    # Ensure that std_val is not zero to avoid division by zero
    if std_val > 0:
        # Perform Z-score normalization
        z_score_data = (tif_data - mean_val) / std_val
        
        # Scale Z-scores to the range 0-255
        scaled_data = 255 * (z_score_data - np.min(z_score_data)) / (np.max(z_score_data) - np.min(z_score_data))
    else:
        # If the standard deviation is zero, return a flat array
        scaled_data = np.full_like(tif_data, 255 if mean_val > 0 else 0)
    
    return scaled_data.astype(np.uint8)


def create_grid_patches(tif_data, patch_size):
    """Create grid patches and find valid patches that contain at least one pixel with value 1."""
    valid_patches = []
    for row in range(0, tif_data.shape[0] - patch_size + 1, patch_size):
        for col in range(0, tif_data.shape[1] - patch_size + 1, patch_size):
            patch = tif_data[row:row + patch_size, col:col + patch_size]
            if np.any(patch == 1):
                valid_patches.append((row, col))
    return valid_patches

def generate_locations(tif_path, patch_size, output_dir):
    """Generate patch locations and save to a text file."""
    with rasterio.open(tif_path) as src:
        tif_data = src.read(1)

    tif_data = preprocess_label_tif(tif_data)
    valid_patches = create_grid_patches(tif_data, patch_size)

    if not valid_patches:
        print(f"No valid patches found for {tif_path}.")
        return

    base_name = os.path.splitext(os.path.basename(tif_path))[0]
    huc_code = base_name.split('_')[0]

    # Determine the output_subdir based on the pattern in the file path
    if "19050302_50" in tif_path:
        output_subdir = f"/projects/bcrm/nathanj/TIFF_data/Alaska/19050302_50/AK_50_Dataset/{huc_code}"
    elif "19060301" in tif_path:
        output_subdir = f"/projects/bcrm/nathanj/TIFF_data/Alaska/19060301/AK_19060301/{huc_code}"
    elif "19060302" in tif_path:
        output_subdir = f"/projects/bcrm/nathanj/TIFF_data/Alaska/19060302/AK_147_Dataset/{huc_code}"
    elif "19080302" in tif_path:
        output_subdir = f"/projects/bcrm/nathanj/TIFF_data/Alaska/19080302/AK_19080302_Dataset/{huc_code}"
    elif "19090101" in tif_path:
        output_subdir = f"/projects/bcrm/nathanj/TIFF_data/Alaska/19090101/AK_19090101_Dataset/{huc_code}"
    elif "19090103" in tif_path:
        output_subdir = f"/projects/bcrm/nathanj/TIFF_data/Alaska/19090103/AK_19090103_Dataset/{huc_code}"
    else:
        output_subdir = os.path.join(os.path.dirname(tif_path), huc_code)

    output_file = os.path.join(output_dir, f"{base_name}_patch_locations.txt")

    with open(output_file, 'w') as f:
        f.write(f"{tif_path}\n")
        f.write(f"{output_subdir}\n")
        for min_row, min_col in valid_patches:
            f.write(f"{min_row},{min_col}\n")

    print(f"Processed {tif_path}:")
    print(f"  Valid patches: {len(valid_patches)}")
    print(f"  Patch locations saved to {output_file}")

def traverse_and_process(data_folder, patch_size, output_dir):
    """Traverse the data folder and process all TIFF files."""
    for root, _, files in os.walk(data_folder):
        for file in files:
            if "_filtered_ref" in file and file.endswith(".tif"):
                tif_path = os.path.join(root, file)
                if "19050302_74" in tif_path:
                    print(f"Skipping {tif_path} as it contains '19050302_74'.")
                    continue
                generate_locations(tif_path, patch_size, output_dir)

def read_patch_locations(file_path):
    """Read the patch locations from a text file."""
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # First line is the TIFF file path
    tif_path = lines[0].strip()

    # Second line is the data subdir path
    data_dir = lines[1].strip()

    # Remaining lines are the patch locations
    patch_locations = [tuple(map(int, line.strip().split(','))) for line in lines[2:]]

    return tif_path, data_dir, patch_locations

def stack_tif_files(data_dir, huc_code):
    """Stack specified TIFF files to create an 8-channel array."""
    if "AK_50_Dataset" in data_dir:
        file_names = [
            f"curvature_{huc_code}.tif",
            f"swm1_{huc_code}.tif",
            f"swm2_{huc_code}.tif",
            f"ori_{huc_code}.tif",  # Adjusted filename
            f"dsm_{huc_code}.tif",
            f"geomorph_{huc_code}.tif",
            f"pos_openness_{huc_code}.tif",
            f"tpi_11_{huc_code}.tif",
            f"twi_{huc_code}.tif"
        ]
    else:
        file_names = [
            f"curvature_{huc_code}.tif",
            f"swm1_{huc_code}.tif",
            f"swm2_{huc_code}.tif",
            f"ori_ave_{huc_code}.tif",
            f"dsm_{huc_code}.tif",
            f"geomorph_{huc_code}.tif",
            f"pos_openness_{huc_code}.tif",
            f"tpi_11_{huc_code}.tif",
            f"twi_{huc_code}.tif"
        ]

    # Read and stack the TIFF files
    channels = []
    for file_name in file_names:
        file_path = os.path.join(data_dir, file_name)
        with rasterio.open(file_path) as src:
            tif_data = src.read(1)
            normalized_data = normalize_tif(tif_data)
            channels.append(normalized_data)

    stacked_array = np.stack(channels, axis=-1)
    return stacked_array

def extract_patches(stacked_array, patch_locations, patch_size):
    """Extract patches from the stacked array using the provided locations."""
    patches = []
    for row, col in patch_locations:
        patch = stacked_array[row:row + patch_size, col:col + patch_size]
        patches.append(patch)
    return patches

def save_patches(patches, output_dir, huc_code, suffix):
    """Save the extracted patches to disk."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    patch_file = os.path.join(output_dir, f"{huc_code}_{suffix}.npy")
    np.save(patch_file, patches)
    print(f"Saved {len(patches)} patches to {patch_file}")

def process_patch_files(patch_files, patch_size, output_dir):
    """Process each patch file to extract and save patches."""
    huc_code_data_patches = {}
    huc_code_label_patches = {}

    for patch_file in patch_files:
        print(f"Processing patch file: {patch_file}")
        tif_path, data_dir, patch_locations = read_patch_locations(patch_file)
        huc_code = os.path.basename(data_dir)  # Extract the HUC code from the directory path
        print(f"  TIF path: {tif_path}")
        print(f"  Data directory: {data_dir}")
        print(f"  HUC code: {huc_code}")
        print(f"  Number of patch locations: {len(patch_locations)}")

        # Stack the TIFF files to create an 8-channel array
        stacked_array = stack_tif_files(data_dir, huc_code)
        print(f"  Stacked array shape: {stacked_array.shape}")

        # Extract patches using the patch locations
        data_patches = extract_patches(stacked_array, patch_locations, patch_size)
        print(f"  Extracted data patches: {len(data_patches)}")

        # Extract label patches from the input TIFF file
        with rasterio.open(tif_path) as src:
            label_array = src.read(1)
            label_array[label_array != 1] = 0
            label_patches = extract_patches(label_array, patch_locations, patch_size)
        print(f"  Extracted label patches: {len(label_patches)}")

        # Accumulate patches by HUC code
        if huc_code not in huc_code_data_patches:
            huc_code_data_patches[huc_code] = []
            huc_code_label_patches[huc_code] = []

        huc_code_data_patches[huc_code].extend(data_patches)
        huc_code_label_patches[huc_code].extend(label_patches)

    # Save the accumulated patches for each HUC code
    for huc_code in huc_code_data_patches:
        print(f"Saving patches for HUC code: {huc_code}")
        save_patches(huc_code_data_patches[huc_code], output_dir, huc_code, 'data')
        save_patches(huc_code_label_patches[huc_code], output_dir, huc_code, 'label')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate patch locations and extract patches from TIFF files.")
    parser.add_argument("--data_folder", type=str, default='/projects/bcrm/nathanj/TIFF_data/Alaska/', required=True, help="Root folder containing the TIFF files")
    parser.add_argument("--patch_size", type=int, default=128, help="Size of the patch (default: 224)")
    parser.add_argument("--output_dir", type=str, default='/u/nathanj/meta-learning-streamline-delineation/scripts/Alaska/data_gen/huc_code_data_znorm_128', required=True, help="Output directory for the text files and patches")

    args = parser.parse_args()



    # Step 1: Traverse and process to generate patch locations
    traverse_and_process(args.data_folder, args.patch_size, args.output_dir)

    # Step 2: Process each patch file to extract and save patches
    patch_files = [os.path.join(args.output_dir, file) for file in os.listdir(args.output_dir) if file.endswith("_patch_locations.txt")]
    process_patch_files(patch_files, args.patch_size, args.output_dir)


# python generate_patch_locations_tifs.py --data_folder "/projects/bcrm/nathanj/TIFF_data/Alaska" --patch_size 224 --output_dir ./huc_code_data_znorm_224/
