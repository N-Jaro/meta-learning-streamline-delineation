import os
import numpy as np
from datetime import datetime
import argparse

def patchify_label_data(label_data_path, patch_save_path, metadata_save_path, img_width=224, buf=30):
    # Load the label data from the .npy file
    label_data = np.load(label_data_path)
    original_shape = label_data.shape
    print(f"Label data shape: {original_shape}")

    # Add symmetric padding
    padded_label_data = np.pad(label_data, ((buf, buf), (buf, buf)), 'symmetric')
    padded_shape = padded_label_data.shape
    print(f"Padded label data shape: {padded_shape}")

    # Calculate effective patch size
    effective_patch_size = img_width - 2 * buf

    # Number of patch rows and columns
    num_patches_row = (padded_shape[0] + effective_patch_size - 1) // effective_patch_size
    num_patches_col = (padded_shape[1] + effective_patch_size - 1) // effective_patch_size

    # Ensure that we count only the valid rows and columns
    valid_num_patches_row = sum(1 for i in range(num_patches_row) if (i * effective_patch_size + img_width) <= padded_shape[0])
    valid_num_patches_col = sum(1 for j in range(num_patches_col) if (j * effective_patch_size + img_width) <= padded_shape[1])

    print('Number of valid rows:', valid_num_patches_row)
    print('Number of valid columns:', valid_num_patches_col)

    # Splitting the total data into patches
    patches = []
    count = 0
    for i in range(num_patches_row):
        for j in range(num_patches_col):
            start_row = i * effective_patch_size
            end_row = start_row + img_width
            start_col = j * effective_patch_size
            end_col = start_col + img_width

            # Skip patches that exceed the original dimensions
            if end_row > padded_shape[0] or end_col > padded_shape[1]:
                continue

            count += 1
            # Extract the patch
            patch = padded_label_data[start_row:end_row, start_col:end_col]
            patches.append(patch[np.newaxis, :, :])

    patches = np.concatenate(patches, axis=0)
    print(patches.shape)  # Should print (number of valid patches, 224, 224)

    # Save the patches and metadata
    np.save(patch_save_path, patches)
    
    # Save metadata
    metadata = {
        "original_shape": original_shape,
        "patch_size": img_width,
        "buffer_size": buf,
        "num_patches_row": valid_num_patches_row,
        "num_patches_col": valid_num_patches_col
    }
    np.savez(metadata_save_path, **metadata)
    
    print(f"Patches saved as {patch_save_path}")
    print(f"Metadata saved as {metadata_save_path}")

def unpatchify(patches_path, metadata_path, reconstructed_save_path):
    # Load patches and metadata
    patches = np.load(patches_path)
    metadata = np.load(metadata_path)
    
    original_shape = metadata['original_shape']
    patch_size = metadata['patch_size']
    buffer_size = metadata['buffer_size']
    num_patches_row = metadata['num_patches_row']
    num_patches_col = metadata['num_patches_col']
    
    effective_patch_size = patch_size - 2 * buffer_size

    # Initialize an empty array for the reconstructed image
    reconstructed_image = np.zeros(original_shape)

    count = 0
    for i in range(num_patches_row):
        for j in range(num_patches_col):
            start_row = i * effective_patch_size
            end_row = start_row + patch_size
            start_col = j * effective_patch_size
            end_col = start_col + patch_size

            reconstructed_image[start_row:end_row, start_col:end_col] = patches[count]
            count += 1

    # Save the reconstructed image
    np.save(reconstructed_save_path, reconstructed_image)
    print(f"Reconstructed image saved as {reconstructed_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Patchify and Unpatchify Scripts")
    parser.add_argument("operation", type=str, choices=["patchify", "unpatchify"], help="Operation to perform")
    parser.add_argument("folder_path", type=str, help="Path to the folder containing the data files")
    
    args = parser.parse_args()

    if args.operation == "patchify":
        label_data_path = os.path.join(args.folder_path, "bottom_half_test_label.npy")
        patch_save_path = os.path.join(args.folder_path, "bottom_half_test_label_patches.npy")
        metadata_save_path = os.path.join(args.folder_path, "bottom_half_test_label_metadata.npz")
        patchify_label_data(label_data_path, patch_save_path, metadata_save_path)
    elif args.operation == "unpatchify":
        patches_path = os.path.join(args.folder_path, "bottom_half_test_label_patches.npy")
        metadata_path = os.path.join(args.folder_path, "bottom_half_test_label_metadata.npz")
        reconstructed_save_path = os.path.join(args.folder_path, "bottom_half_test_label_reconstructed.npy")
        unpatchify(patches_path, metadata_path, reconstructed_save_path)


# To patchify:
# python script.py patchify "/u/nathanj/meta_learning/samples/<location>/"

# To unpatchify:
# python script.py unpatchify "/u/nathanj/meta_learning/samples/<location>/"