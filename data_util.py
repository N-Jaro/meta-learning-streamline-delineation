import os
import numpy as np
import tensorflow as tf

class MetaDataLoader:
    def __init__(self, data_dir, num_samples_per_location=100, normalization_type='-1'):
        self.data_dir = data_dir
        self.num_samples_per_location = num_samples_per_location
        self.normalization_type = normalization_type
        self.data_dict = {}  # This dictionary will cache the loaded data.

    def load_all_data(self, locations):
        """Loads all necessary data for the given locations and caches it in memory."""
        for location in locations:
            location_dir = os.path.join(self.data_dir, location)

            # File paths for training and validation data and labels
            train_data_path = os.path.join(location_dir, "train_data.npy")
            train_label_path = os.path.join(location_dir, "train_label.npy")
            vali_data_path = os.path.join(location_dir, "vali_data.npy")
            vali_label_path = os.path.join(location_dir, "vali_label.npy")

            # Load data
            train_data = np.load(train_data_path)
            vali_data = np.load(vali_data_path)

            # Ensure no negative values
            train_data[train_data < 0] = 0
            vali_data[vali_data < 0] = 0

            # Normalize data
            train_data = self.normalize_data(train_data)
            vali_data = self.normalize_data(vali_data)

            # Store the processed data and labels in the data_dict
            self.data_dict[location] = {
                'train_data': train_data,
                'train_label': np.load(train_label_path),
                'vali_data': vali_data,
                'vali_label': np.load(vali_label_path)
            }

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
        else:
            raise ValueError("Unsupported normalization type. Choose '0-1' or '-1-1'.")

    def _create_episode(self, location):
        # Randomly select samples from one location for both support and query sets
        support_set_data, support_set_labels = self._random_select_samples(location, 'train')
        query_set_data, query_set_labels = self._random_select_samples(location, 'vali')

        return support_set_data, support_set_labels, query_set_data, query_set_labels

    def _random_select_samples(self, location, data_type):
        data = self.data_dict[location][f'{data_type}_data']
        labels = self.data_dict[location][f'{data_type}_label']

        if len(data) >= self.num_samples_per_location:
            indices = np.random.choice(len(data), self.num_samples_per_location, replace=False)
            selected_data = data[indices]
            selected_labels = labels[indices]
        else:
            print(f"Warning: Not enough samples in {location}")
            selected_data = np.array([])
            selected_labels = np.array([])

        return selected_data, selected_labels

    def create_multi_episodes(self, num_episodes, locations):
        self.load_all_data(locations)  # Load data once here
        episodes = []
        for _ in range(num_episodes):
            location = np.random.choice(locations)  # Select a random location
            print(location)  # For debugging, you might want to remove or comment this out in production
            episode_data = self._create_episode(location)
            episode = {
                "support_set_data": episode_data[0],
                "support_set_labels": episode_data[1],
                "query_set_data": episode_data[2],
                "query_set_labels": episode_data[3]
            }
            episodes.append(episode)
        return episodes
