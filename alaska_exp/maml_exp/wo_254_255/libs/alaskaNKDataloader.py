import tensorflow as tf
import numpy as np
import os
import pandas as pd

class AlaskaNKMetaDataset:
    def __init__(self, data_dir, csv_file, normalization_type='none', transform=None, channels=None, verbose=True):
        """
        Args:
            data_dir (string): Directory with all the .npy files.
            csv_file (string): Path to the CSV file containing huc_codes.
            normalization_type (string): Type of normalization ('0', '-1', 'none').
            transform (callable, optional): Optional transform to be applied on a sample.
            channels (list of int): List of channels to select from the data.
            verbose (bool): If True, print progress and debug information.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.normalization_type = normalization_type
        self.channels = channels if channels is not None else list(range(9))  # Default to all 9 channels if not specified
        self.verbose = verbose
        self.data = {}
        self.labels = {}
        self.huc_codes = self.load_huc_codes_from_csv(csv_file)
        self.min_samples = None
        self.episode_record = []  # To track watersheds used in each episode
        if self.verbose:
            print(f"HUC codes loaded: {self.huc_codes}")
        
    def load_huc_codes_from_csv(self, csv_file):
        """
        Load HUC codes from the given CSV file.
        """
        df = pd.read_csv(csv_file)
        return df['huc_code'].tolist()
    
    def load_all_data(self):
        """
        Load all data for the given huc_codes and store in memory. Also find the least number of samples.
        """
        min_samples = float('inf')
        for huc_code in self.huc_codes:
            data_path = os.path.join(self.data_dir, f"{huc_code}_data.npy")
            label_path = os.path.join(self.data_dir, f"{huc_code}_label.npy")
            
            if self.verbose:
                print(f"Loading data for HUC code {huc_code}...")
            data = np.load(data_path)
            labels = np.load(label_path)
            if self.verbose:
                print(f"Loaded {data.shape[0]} samples.")
            
            # Select specified channels
            data = data[..., self.channels]
            
            self.data[huc_code] = data
            self.labels[huc_code] = labels
            
            num_samples = data.shape[0]
            if num_samples < min_samples:
                min_samples = num_samples
        
        self.min_samples = min_samples
        if self.verbose:
            print(f"Minimum number of samples across all HUC codes: {self.min_samples}")
    
    def normalize_data(self, data):
        """Normalizes data based on the specified normalization type."""
        if self.normalization_type == '0':
            data_min = 0
            data_max = 255
            return (data - data_min) / (data_max - data_min)
        elif self.normalization_type == '-1':
            data_min = 0
            data_max = 255
            return 2 * ((data - data_min) / (data_max - data_min)) - 1
        elif self.normalization_type == 'none':
            return data
        else:
            raise ValueError("Unsupported normalization type. Choose '0', '-1', or 'none'.")
    
    def _create_episode(self, N, K):
        """
        Create a single N-way K-shot episode.
        """
        if self.verbose:
            print(f"Creating an episode with {N}-way {K}-shot...")
        selected_huc_codes = np.random.choice(self.huc_codes, N, replace=False)
        if self.verbose:
            print(f"Selected HUC codes for the episode: {selected_huc_codes}")

        # Record selected HUC codes for this episode
        self.episode_record.append(selected_huc_codes)
        
        support_set_data = []
        support_set_labels = []
        query_set_data = []
        query_set_labels = []
        
        for huc_code in selected_huc_codes:
            data = self.data[huc_code]
            labels = self.labels[huc_code]
            
            N_samples = data.shape[0]
            indices = np.arange(N_samples)
            selected_indices = np.random.choice(indices, 2 * K, replace=False)
            
            support_indices = selected_indices[:K]
            query_indices = selected_indices[K:]
            
            support_set_data.append(self.normalize_data(data[support_indices]))
            support_set_labels.append(labels[support_indices])
            query_set_data.append(self.normalize_data(data[query_indices]))
            query_set_labels.append(labels[query_indices])
        
        support_set_data = np.concatenate(support_set_data, axis=0)
        support_set_labels = np.concatenate(support_set_labels, axis=0)
        query_set_data = np.concatenate(query_set_data, axis=0)
        query_set_labels = np.concatenate(query_set_labels, axis=0)
        
        if self.transform:
            support_set_data = self.transform(support_set_data)
            support_set_labels = self.transform(support_set_labels)
            query_set_data = self.transform(query_set_data)
            query_set_labels = self.transform(query_set_labels)
        
        if self.verbose:
            print(f"Episode created with {support_set_data.shape[0]} support samples and {query_set_data.shape[0]} query samples.")
        return support_set_data, support_set_labels, query_set_data, query_set_labels

    def create_multi_episodes(self, num_episodes, N, K):
        """
        Create multiple N-way K-shot episodes for meta-learning.
        """
        self.load_all_data()  # Load data once
        
        # Ensure K is valid
        if K > self.min_samples // 2:
            raise ValueError(f"K should be less than or equal to {self.min_samples // 2}")
        
        episodes = []
        print(f"Creating {num_episodes} episodes...")
        for i in range(num_episodes):
            if self.verbose:
                print(f"Creating episode {i+1}/{num_episodes}...")
            episode_data = self._create_episode(N, K)
            episode = {
                "support_set_data": tf.convert_to_tensor(episode_data[0], dtype=tf.float32),
                "support_set_labels": tf.convert_to_tensor(episode_data[1], dtype=tf.float32),
                "query_set_data": tf.convert_to_tensor(episode_data[2], dtype=tf.float32),
                "query_set_labels": tf.convert_to_tensor(episode_data[3], dtype=tf.float32)
            }
            episodes.append(episode)
        print(f"Finished creating {num_episodes} episodes.")
        return episodes
        
    
    def get_episode_record(self):
        """Returns the record of watersheds (HUC codes) used in each episode."""
        return self.episode_record

# # Example usage
# if __name__ == "__main__":
#     csv_file = 'path_to_huc_codes.csv'  # Path to your CSV file
#     selected_channels = [0, 1, 2, 3, 4, 5, 6, 7]  # Specify the 8 channels you want to select
#     dataset = AlaskaMetaDataset(data_dir='path_to_npy_files', csv_file=csv_file, normalization_type='0', channels=selected_channels, verbose=True)
#     episodes = dataset.create_multi_episodes(num_episodes=5, N=3, K=5)
    
#     for i, episode in enumerate(episodes):
#         print(f"Episode {i+1}:")
#         print(f"  Support Set - Data shape: {episode['support_set_data'].shape}, Labels shape: {episode['support_set_labels'].shape}")
#         print(f"  Query Set - Data shape: {episode['query_set_data'].shape}, Labels shape: {episode['query_set_labels'].shape}")
    
#     # Get the record of watersheds (HUC codes) used in each episode
#     episode_record = dataset.get_episode_record()
#     print("Record of watersheds used in each episode:", episode_record)