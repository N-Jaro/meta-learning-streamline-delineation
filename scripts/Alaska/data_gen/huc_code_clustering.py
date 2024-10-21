import os
import argparse
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import random

def load_patches(data_dir, huc_code):
    """Load data and label patches for a given HUC code."""
    data_path = os.path.join(data_dir, f"{huc_code}_data.npy")
    label_path = os.path.join(data_dir, f"{huc_code}_label.npy")
    
    data_patches = np.load(data_path)
    label_patches = np.load(label_path)
    
    return data_patches, label_patches

def get_stream_pixels(data_patches, label_patches):
    """Get the data of stream pixels from the patches."""
    stream_pixels = []
    
    for patch_data, patch_label in zip(data_patches, label_patches):
        stream_pixel_indices = np.argwhere(patch_label == 1)
        for idx in stream_pixel_indices:
            pixel_data = patch_data[idx[0], idx[1], :]
            stream_pixels.append(pixel_data)
    
    return np.array(stream_pixels)

def select_random_stream_pixels(stream_pixels, num_samples=10):
    """Select random stream pixels."""
    if len(stream_pixels) <= num_samples:
        return stream_pixels
    else:
        return stream_pixels[random.sample(range(len(stream_pixels)), num_samples)]

def main(input_folder, num_clusters, output_path):
    all_stream_pixels = []
    huc_codes = []
    
    # Traverse input folder to find _data.npy and _label.npy files
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith("_data.npy"):
                huc_code = file.split("_data.npy")[0]
                data_patches, label_patches = load_patches(root, huc_code)
                
                stream_pixels = get_stream_pixels(data_patches, label_patches)
                selected_stream_pixels = select_random_stream_pixels(stream_pixels)
                
                all_stream_pixels.extend(selected_stream_pixels)
                huc_codes.extend([huc_code] * len(selected_stream_pixels))
    
    # Perform K-means clustering on the combined stream pixels
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(all_stream_pixels)
    
    # Prepare the output dataframe
    output_df = pd.DataFrame({'huc_code': huc_codes, 'cluster': kmeans.labels_})
    
    # Save to CSV
    output_df.to_csv(output_path, index=False)
    print(f"Cluster results saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="K-means clustering of stream pixels from HUC codes.")
    parser.add_argument("--input_folder", type=str, required=True, help="Input folder containing the _data.npy and _label.npy files")
    parser.add_argument("--num_clusters", type=int, required=True, help="Number of clusters for K-means")
    parser.add_argument("--output_path", type=str, required=True, help="Output path to save the CSV file")
    
    args = parser.parse_args()
    
    main(args.input_folder, args.num_clusters, args.output_path)
