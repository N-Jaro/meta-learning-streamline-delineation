import os
import random
import rasterio
import numpy as np
from rtree import index

def generate_samples(tif_path, num_train_samples, num_val_samples, patch_size, output_dir):
    patch_size_w, patch_size_h = patch_size, patch_size

    with rasterio.open(tif_path) as src:
        array = src.read(1)  # Read the first band
        rows, cols = array.shape
        print("Image Size:", rows, cols)

    def is_valid_patch(array, min_row, min_col, patch_size_w, patch_size_h):
        patch = array[min_row:min_row + patch_size_w, min_col:min_col + patch_size_h]
        return np.sum(patch == 1) >= 50  # At least 50 pixels of 1

    def generate_patch_locations(num_samples, exclude_idx, array, patch_size_w, patch_size_h):
        num_attemps = 30000
        locations = []
        idx = index.Index()
        
        for i in range(num_samples):
            attempts = 0
            while attempts < num_attemps:  # Limit number of attempts to avoid infinite loop
                min_row = random.randint(0, rows - patch_size_w)
                min_col = random.randint(0, cols - patch_size_h)
                if is_valid_patch(array, min_row, min_col, patch_size_w, patch_size_h):
                    bbox = (min_col, min_row, min_col + patch_size_w, min_row + patch_size_h)
                    if not list(idx.intersection(bbox)) and (not exclude_idx or not list(exclude_idx.intersection(bbox))):
                        locations.append((min_row, min_col))
                        idx.insert(i, bbox)
                        break
                attempts += 1
            if attempts == num_attemps:
                return None, False
        return locations, True

    train_locations, train_success = generate_patch_locations(num_train_samples, None, array, patch_size_w, patch_size_h)
    if not train_success:
        print("Not enough valid training samples could be generated.")
        return

    train_idx = index.Index()
    for i, (min_row, min_col) in enumerate(train_locations):
        bbox = (min_col, min_row, min_col + patch_size_w, min_row + patch_size_h)
        train_idx.insert(i, bbox)

    val_locations, val_success = generate_patch_locations(num_val_samples, train_idx, array, patch_size_w, patch_size_h)
    if not val_success:
        print("Not enough valid validation samples could be generated.")
        return

    tif_name = os.path.splitext(os.path.basename(tif_path))[0]
    
    train_output_path = os.path.join(output_dir, f"{tif_name}_train_{patch_size_w}x{patch_size_h}.txt")
    val_output_path = os.path.join(output_dir, f"{tif_name}_val_{patch_size_w}x{patch_size_h}.txt")

    def save_locations(file_path, locations):
        with open(file_path, 'w') as file:
            for min_row, min_col in locations:
                file.write(f"{min_row},{min_col}\n")

    save_locations(train_output_path, train_locations)
    save_locations(val_output_path, val_locations)
    print(f"Training samples saved to: {train_output_path}")
    print(f"Validation samples saved to: {val_output_path}")

# Example usage:
# generate_samples('path/to/tif_file.tif', 100, 20, 224, 'output/directory')
