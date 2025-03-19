import os
import argparse
import numpy as np
import rasterio
import rasterio.mask
from rasterio.windows import Window

def normalize_tif(tif_data):
    """Normalize the TIFF data using mean and standard deviation, then scale to 0-255 range."""
    tif_data[tif_data == -9999.0] = 0  # Handle NoData values
    mean_val = np.mean(tif_data)
    std_val = np.std(tif_data)

    if std_val > 0:
        # Perform Z-score normalization
        z_score_data = (tif_data - mean_val) / std_val
        # Scale Z-scores to 0-255
        scaled_data = 255 * (z_score_data - np.min(z_score_data)) / (np.max(z_score_data) - np.min(z_score_data))
    else:
        # If no variance, return a flat array
        scaled_data = np.full_like(tif_data, 255 if mean_val > 0 else 0)
    
    return scaled_data.astype(np.uint8)

def pad_array(tif_data, patch_size):
    """Pad the array to ensure it is divisible by the patch size."""
    h, w = tif_data.shape
    pad_h = (patch_size - (h % patch_size)) % patch_size
    pad_w = (patch_size - (w % patch_size)) % patch_size
    
    padded_data = np.pad(tif_data, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
    return padded_data, (h, w)  # Return original size for later reconstruction

def extract_patches_with_padding(tif_data, patch_size):
    """Extract patches of the given size with padding if needed."""
    padded_data, original_shape = pad_array(tif_data, patch_size)
    patches = []
    locations = []

    for row in range(0, padded_data.shape[0], patch_size):
        for col in range(0, padded_data.shape[1], patch_size):
            patch = padded_data[row:row + patch_size, col:col + patch_size]
            patches.append(patch)
            locations.append((row, col))

    return np.array(patches), locations, original_shape

def stack_huc_tif_files(data_dir, huc_code, patch_size):
    """Stack all relevant TIFF files for a given HUC code and extract patches."""
    file_names = [
        f"curvature_{huc_code}.tif",
        f"swm1_{huc_code}.tif",
        f"swm2_{huc_code}.tif",
        f"ori_ave_{huc_code}.tif",
        f"dsm_{huc_code}.tif",
        f"geomorph_{huc_code}.tif",
        f"pos_openness_{huc_code}.tif",
        f"tpi_11_{huc_code}.tif",
        f"twi_{huc_code}.tif",
        f"tpi_3_{huc_code}.tif",
        f"dtm_{huc_code}.tif",
    ]

    channels = []
    for file_name in file_names:
        file_path = os.path.join(data_dir, file_name)
        if not os.path.exists(file_path):
            print(f"Warning: {file_name} not found, skipping.")
            continue
        
        with rasterio.open(file_path) as src:
            tif_data = src.read(1)
            normalized_data = normalize_tif(tif_data)
            channels.append(normalized_data)

    if not channels:
        raise ValueError(f"No valid TIFF files found for HUC code {huc_code}")

    stacked_array = np.stack(channels, axis=-1)
    
    # Extract patches from stacked array
    patches, locations, original_shape = extract_patches_with_padding(stacked_array, patch_size)

    return patches, locations, original_shape

def save_patches(patches, output_dir, huc_code):
    """Save the extracted patches as .npy files."""
    os.makedirs(output_dir, exist_ok=True)
    patch_file = os.path.join(output_dir, f"{huc_code}_patches.npy")
    np.save(patch_file, patches)
    print(f"Saved {patches.shape[0]} patches to {patch_file}")

def reconstruct_from_patches(patch_predictions, locations, original_shape, patch_size):
    """Reconstruct the HUC code-sized prediction from patches."""
    padded_h, padded_w = original_shape[0] + (patch_size - original_shape[0] % patch_size) % patch_size, \
                         original_shape[1] + (patch_size - original_shape[1] % patch_size) % patch_size

    reconstructed = np.zeros((padded_h, padded_w), dtype=np.float32)
    count_map = np.zeros((padded_h, padded_w), dtype=np.int32)

    for patch, (row, col) in zip(patch_predictions, locations):
        reconstructed[row:row + patch_size, col:col + patch_size] += patch
        count_map[row:row + patch_size, col:col + patch_size] += 1

    # Avoid division by zero and restore original size
    count_map[count_map == 0] = 1
    reconstructed /= count_map
    return reconstructed[:original_shape[0], :original_shape[1]]

def save_reconstructed_tif(output_dir, huc_code, reference_tif, reconstructed_array):
    """Save the reconstructed prediction array as a GeoTIFF."""
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{huc_code}_prediction.tif")

    with rasterio.open(reference_tif) as src:
        meta = src.meta.copy()
        meta.update(dtype=rasterio.float32, count=1)

        with rasterio.open(output_file, 'w', **meta) as dst:
            dst.write(reconstructed_array, 1)
    
    print(f"Reconstructed result saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract patches from a single HUC code and reconstruct predictions.")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing TIFF files for the HUC code")
    parser.add_argument("--huc_code", type=str, required=True, help="HUC code to process")
    parser.add_argument("--patch_size", type=int, default=128, help="Patch size (default: 128)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save extracted patches and reconstructed results")
    parser.add_argument("--prediction_file", type=str, required=False, help="(Optional) Prediction file to reconstruct")
    parser.add_argument("--reference_tif", type=str, required=False, help="(Optional) Reference TIFF file for georeferencing output")

    args = parser.parse_args()

    # Extract patches
    patches, locations, original_shape = stack_huc_tif_files(args.data_dir, args.huc_code, args.patch_size)
    save_patches(patches, args.output_dir, args.huc_code)

    # If a prediction file is provided, reconstruct and save results
    if args.prediction_file and args.reference_tif:
        prediction_patches = np.load(args.prediction_file)
        reconstructed_array = reconstruct_from_patches(prediction_patches, locations, original_shape, args.patch_size)
        save_reconstructed_tif(args.output_dir, args.huc_code, args.reference_tif, reconstructed_array)

