import os
import numpy as np
import tensorflow as tf

class MetaDataLoader:
    def __init__(self, data_dir, num_samples_per_location=100, stream_pixel_per_patch=10):
        self.data_dir = data_dir
        self.num_samples_per_location = num_samples_per_location
        self.stream_pixel_per_patch = stream_pixel_per_patch
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

            # Load and store the data and labels in the data_dict
            self.data_dict[location] = {
                'train_data': np.load(train_data_path),
                'train_label': np.load(train_label_path),
                'vali_data': np.load(vali_data_path),
                'vali_label': np.load(vali_label_path)
            }

    def _create_episode(self, locations):
        selected_data = []
        selected_labels = []

        # Processing for creating support set
        for location in locations:
            temp_data = []
            temp_labels = []

            # Iterate over samples in the location's training data
            for data, label in zip(self.data_dict[location]['train_data'], self.data_dict[location]['train_label']):
                if np.sum(label > 0) >= self.stream_pixel_per_patch:
                    temp_data.append(data)
                    temp_labels.append(label)

            if len(temp_data) >= self.num_samples_per_location:
                indices = np.random.choice(len(temp_data), self.num_samples_per_location, replace=False)
                selected_data.extend(np.array(temp_data)[indices])
                selected_labels.extend(np.array(temp_labels)[indices])
            else:
                print(f"Warning: Not enough samples in {location}")

        support_set_data = np.array(selected_data)
        support_set_labels = np.array(selected_labels)

        selected_data = []
        selected_labels = []

        # Processing for creating query set
        for location in locations:
            temp_data = []
            temp_labels = []

            for data, label in zip(self.data_dict[location]['vali_data'], self.data_dict[location]['vali_label']):
                if np.sum(label > 0) >= self.stream_pixel_per_patch:
                    temp_data.append(data)
                    temp_labels.append(label)

            if len(temp_data) >= self.num_samples_per_location:
                indices = np.random.choice(len(temp_data), self.num_samples_per_location, replace=False)
                selected_data.extend(np.array(temp_data)[indices])
                selected_labels.extend(np.array(temp_labels)[indices])
            else:
                print(f"Warning: Not enough samples in {location}")

        query_set_data = np.array(selected_data)
        query_set_labels = np.array(selected_labels)

        return support_set_data, support_set_labels, query_set_data, query_set_labels

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
