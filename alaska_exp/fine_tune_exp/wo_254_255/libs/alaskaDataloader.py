import torch
from torch.utils.data import Dataset
import numpy as np
import os
import pandas as pd

class AlaskaMetaDataset:
    def __init__(self, data_dir, csv_file, transform=None):
        """
        Args:
            data_dir (string): Directory with all the .npy files.
            csv_file (string): Path to the CSV file containing huc_codes.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.data = {}
        self.labels = {}
        self.huc_codes = self.load_huc_codes_from_csv(csv_file)
        self.min_samples = None
        
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
            
            data = np.load(data_path)
            labels = np.load(label_path)
            
            self.data[huc_code] = data
            self.labels[huc_code] = labels
            
            num_samples = data.shape[0]
            if num_samples < min_samples:
                min_samples = num_samples
        
        self.min_samples = min_samples

    def _create_episode(self, num_samples_per_huc_code, huc_code):
        """
        Create a single episode from the specified huc_code.
        """
        data = self.data[huc_code]
        labels = self.labels[huc_code]
        
        N = data.shape[0]
        indices = np.arange(N)
        selected_indices = np.random.choice(indices, 2 * num_samples_per_huc_code, replace=False)
        
        support_indices = selected_indices[:num_samples_per_huc_code]
        query_indices = selected_indices[num_samples_per_huc_code:]
        
        support_set_data = data[support_indices]
        support_set_labels = labels[support_indices]
        query_set_data = data[query_indices]
        query_set_labels = labels[query_indices]
        
        if self.transform:
            support_set_data = self.transform(support_set_data)
            support_set_labels = self.transform(support_set_labels)
            query_set_data = self.transform(query_set_data)
            query_set_labels = self.transform(query_set_labels)
        
        return support_set_data, support_set_labels, query_set_data, query_set_labels

    def create_multi_episodes(self, num_samples_per_huc_code):
        """
        Create multiple episodes for meta-learning.
        """
        self.load_all_data()  # Load data once
        print(f"Minimum number of samples across all HUC codes: {self.min_samples}")
        
        # Ensure num_samples_per_huc_code is valid
        if num_samples_per_huc_code > self.min_samples // 2:
            raise ValueError(f"num_samples_per_huc_code should be less than or equal to {self.min_samples // 2}")
        
        episodes = []
        for huc_code in self.huc_codes:
            episode_data = self._create_episode(num_samples_per_huc_code, huc_code)
            episode = {
                "support_set_data": torch.tensor(episode_data[0], dtype=torch.float32),
                "support_set_labels": torch.tensor(episode_data[1], dtype=torch.long),
                "query_set_data": torch.tensor(episode_data[2], dtype=torch.float32),
                "query_set_labels": torch.tensor(episode_data[3], dtype=torch.long)
            }
            episodes.append(episode)
        return episodes

# # Example usage
# if __name__ == "__main__":
#     csv_file = 'path_to_huc_codes.csv'  # Path to your CSV file
#     dataset = AlaskaMetaDataset(data_dir='path_to_npy_files', csv_file=csv_file)
#     episodes = dataset.create_multi_episodes(num_samples_per_huc_code=10)
    
#     for i, episode in enumerate(episodes):
#         print(f"Episode {i+1}:")
#         print(f"  Support Set - Data shape: {episode['support_set_data'].shape}, Labels shape: {episode['support_set_labels'].shape}")
#         print(f"  Query Set - Data shape: {episode['query_set_data'].shape}, Labels shape: {episode['query_set_labels'].shape}")
