import os
import numpy as np
import tensorflow as tf

class MetaDataLoader:
    def __init__(self, data_dir, num_samples_per_location=100):
        self.data_dir = data_dir
        self.num_samples_per_location = num_samples_per_location
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

            # Load and store the data and labels in the data_dict, ensuring no negative values
            self.data_dict[location] = {
                'train_data': np.clip(np.load(train_data_path), 0, None),
                'train_label': np.clip(np.load(train_label_path), 0, None),
                'vali_data': np.clip(np.load(vali_data_path), 0, None),
                'vali_label': np.clip(np.load(vali_label_path), 0, None)
            }

    def _create_episode(self, locations):
        # Create support and query sets by randomly selecting samples
        support_set_data, support_set_labels = self._random_select_samples(locations, 'train')
        query_set_data, query_set_labels = self._random_select_samples(locations, 'vali')

        return support_set_data, support_set_labels, query_set_data, query_set_labels

    def _random_select_samples(self, locations, data_type):
        selected_data = []
        selected_labels = []

        for location in locations:
            data = self.data_dict[location][f'{data_type}_data']
            labels = self.data_dict[location][f'{data_type}_label']

            if len(data) >= self.num_samples_per_location:
                indices = np.random.choice(len(data), self.num_samples_per_location, replace=False)
                selected_data.extend(data[indices])
                selected_labels.extend(labels[indices])
            else:
                print(f"Warning: Not enough samples in {location}")

        return np.array(selected_data), np.array(selected_labels)

    def create_multi_episodes(self, num_episodes, locations):
        self.load_all_data(locations)  # Load data once here
        episodes = []
        for _ in range(num_episodes):
            episode_data = self._create_episode(locations)
            episode = {
                "support_set_data": episode_data[0],
                "support_set_labels": episode_data[1],
                "query_set_data": episode_data[2],
                "query_set_labels": episode_data[3]
            }
            episodes.append(episode)
        return episodes
