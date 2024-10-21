import numpy as np
import os
import random
import re
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def load_npy_files(huc_code, data_dir):
    print(f"Loading files for huc_code: {huc_code}")
    label_file = os.path.join(data_dir, f"{huc_code}_label.npy")
    data_file = os.path.join(data_dir, f"{huc_code}_data.npy")
    
    labels = np.load(label_file)  # Shape: (N, 128, 128)
    data = np.load(data_file)     # Shape: (N, 128, 128, 9)
    
    print(f"Loaded {labels.shape[0]} label patches and {data.shape[0]} data patches.")
    return labels, data

def sample_data_points(huc_code, labels, data, num_samples=10):
    print(f"Sampling data points for huc_code: {huc_code}")
    indices_label_1 = np.argwhere(labels == 1)
    indices_label_0 = np.argwhere(labels == 0)
    
    print(f"Found {len(indices_label_1)} points with label 1 and {len(indices_label_0)} points with label 0.")
    
    sampled_label_1_indices = random.sample(list(indices_label_1), min(num_samples, len(indices_label_1)))
    sampled_label_0_indices = random.sample(list(indices_label_0), min(num_samples, len(indices_label_0)))
    
    print(f"Selected {len(sampled_label_1_indices)} samples for label 1 and {len(sampled_label_0_indices)} samples for label 0.")
    
    sampled_data = []
    
    for idx in sampled_label_1_indices + sampled_label_0_indices:
        n, i, j = idx  # Unpack the index (N, 128, 128)
        sample = [huc_code, labels[n, i, j]] + list(data[n, i, j, :])  # Add huc_code, label, and channels 1-9
        sampled_data.append(sample)
    
    return np.array(sampled_data)

def process_all_huc_codes_in_folder(data_dir, num_samples=10):
    print(f"Scanning directory: {data_dir}")
    files = os.listdir(data_dir)
    huc_codes = set(re.match(r"(.+)_label\.npy", f).group(1) for f in files if '_label.npy' in f)
    
    print(f"Found {len(huc_codes)} huc_codes: {list(huc_codes)}")
    
    all_sampled_data = []
    
    for huc_code in huc_codes:
        print(f"\nProcessing huc_code: {huc_code}")
        labels, data = load_npy_files(huc_code, data_dir)
        sampled_data = sample_data_points(huc_code, labels, data, num_samples)
        all_sampled_data.append(sampled_data)
    
    return np.vstack(all_sampled_data)


def create_pca_tsne_plots(data, output_dir):
    # Extract features (channels 1-9) and labels
    features = data[:, 2:].astype(float)  # Channels 1-9
    labels = data[:, 1]  # Extract labels

    # Handle cases where labels might be floats or strings, convert them to int safely
    labels = labels.astype(float).astype(int)
    
    # Perform PCA
    print("Performing PCA...")
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(features)
    
    # Perform t-SNE
    print("Performing t-SNE...")
    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(features)
    
    # Plot PCA
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("PCA of Sampled Data")
    plt.scatter(pca_result[labels == 1, 0], pca_result[labels == 1, 1], label='Label 1', alpha=0.3)
    plt.scatter(pca_result[labels == 0, 0], pca_result[labels == 0, 1], label='Label 0', alpha=0.3)
    plt.legend()
    
    # Plot t-SNE
    plt.subplot(1, 2, 2)
    plt.title("t-SNE of Sampled Data")
    plt.scatter(tsne_result[labels == 1, 0], tsne_result[labels == 1, 1], label='Label 1', alpha=0.3)
    plt.scatter(tsne_result[labels == 0, 0], tsne_result[labels == 0, 1], label='Label 0', alpha=0.3)
    plt.legend()
    
    # Save the plot as a PNG file
    plt.tight_layout()
    pca_tsne_path = os.path.join(output_dir, 'pca_tsne_plots.png')
    plt.savefig(pca_tsne_path, format='png')
    print(f"PCA and t-SNE plots saved as {pca_tsne_path}")


# Example usage:
data_dir = '/u/nathanj/meta-learning-streamline-delineation/scripts/Alaska/data_gen/huc_code_data_znorm_128'      # Replace with your actual directory
output_dir = './'   # Replace with your desired output directory
os.makedirs(output_dir, exist_ok=True)

output_data = process_all_huc_codes_in_folder(data_dir)

# Create PCA and t-SNE plots and save as PNG
create_pca_tsne_plots(output_data, output_dir)
