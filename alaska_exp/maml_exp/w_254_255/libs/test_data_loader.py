import os
import numpy as np

# Assuming the AlaskaMetaDataset class is already defined in a file named alaska_meta_dataset.py
from alaskaNKDataloader import AlaskaNKMetaDataset

def create_dummy_data(data_dir, huc_codes, num_samples):
    """
    Create dummy data for testing purposes.
    """
    os.makedirs(data_dir, exist_ok=True)
    
    for huc_code in huc_codes:
        data = np.random.rand(num_samples, 3, 64, 64)  # Example: 3-channel 64x64 images
        labels = np.random.randint(0, 2, size=(num_samples,))  # Example: binary labels
        
        data_path = os.path.join(data_dir, f"{huc_code}_data.npy")
        label_path = os.path.join(data_dir, f"{huc_code}_label.npy")
        
        np.save(data_path, data)
        np.save(label_path, labels)

def main():
    data_dir = 'test_data'  # Directory to store dummy .npy files
    csv_file = 'huc_codes.csv'  # Path to the CSV file containing huc_codes
    
    # Create dummy HUC codes and corresponding CSV file
    huc_codes = [f"huc{i:03}" for i in range(10)]
    num_samples = 50  # Number of samples per HUC code
    csv_content = "huc_code,exclude_train_test\n" + "\n".join([f"{code},train" for code in huc_codes])
    
    with open(csv_file, 'w') as f:
        f.write(csv_content)
    
    # Create dummy .npy data files
    create_dummy_data(data_dir, huc_codes, num_samples)
    
    # Initialize the AlaskaMetaDataset
    dataset = AlaskaMetaDataset(data_dir=data_dir, csv_file=csv_file)
    
    # Create episodes
    num_episodes = 5
    N = 3  # N-way
    K = 5  # K-shot
    episodes = dataset.create_multi_episodes(num_episodes=num_episodes, N=N, K=K)
    
    # Print details of each episode
    for i, episode in enumerate(episodes):
        print(f"Episode {i+1}:")
        print(f"  Support Set - Data shape: {episode['support_set_data'].shape}, Labels shape: {episode['support_set_labels'].shape}")
        print(f"  Query Set - Data shape: {episode['query_set_data'].shape}, Labels shape: {episode['query_set_labels'].shape}")

if __name__ == "__main__":
    main()
