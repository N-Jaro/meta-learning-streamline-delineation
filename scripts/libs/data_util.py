import os
import numpy as np
import tensorflow as tf
import random

class MetaDataLoader:
    def __init__(self, data_dir, normalization_type='-1'):
        self.data_dir = data_dir
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
        elif self.normalization_type == 'none':
            return data
        else:
            raise ValueError("Unsupported normalization type. Choose '0-1' or '-1-1'.")

    def _create_episode(self, num_samples_per_location, location):
        # Randomly select samples from one location for both support and query sets
        support_set_data, support_set_labels = self._random_select_samples(num_samples_per_location, location, 'train')
        query_set_data, query_set_labels = self._random_select_samples(num_samples_per_location, location, 'vali')

        return support_set_data, support_set_labels, query_set_data, query_set_labels

    def _random_select_samples(self, num_samples_per_location, location, data_type):
        data = self.data_dict[location][f'{data_type}_data']
        labels = self.data_dict[location][f'{data_type}_label']

        if len(data) >= num_samples_per_location:
            indices = np.random.choice(len(data), num_samples_per_location, replace=False)
            selected_data = data[indices]
            selected_labels = labels[indices]
        else:
            print(f"Warning: Not enough samples in {location}")
            selected_data = np.array([])
            selected_labels = np.array([])

        return selected_data, selected_labels

    def create_multi_episodes(self, num_episodes, num_samples_per_location, locations):
        self.load_all_data(locations)  # Load data once here
        episodes = []
        for _ in range(num_episodes):
            location = np.random.choice(locations)  # Select a random location
            print(location)  # For debugging, you might want to remove or comment this out in production
            episode_data = self._create_episode(num_samples_per_location, location)
            episode = {
                "support_set_data": episode_data[0],
                "support_set_labels": episode_data[1],
                "query_set_data": episode_data[2],
                "query_set_labels": episode_data[3]
            }
            episodes.append(episode)
        return episodes



class JointDataLoader:
    def __init__(self, data_path, num_samples, batch_size=32, mode='train'):
        """
        Initialize the JointDataLoader with the path to data, the number of samples to load for training,
        and the mode (train or test).
        
        :param data_path: str, path to the main data directory
        :param num_samples: int, number of samples to load for training (30% of this will be used for validation)
        :param mode: str, either 'train' or 'test', specifying the mode of operation
        """
        self.batch_size = batch_size 
        self.data_path = data_path
        self.num_samples = num_samples
        self.mode = mode
        self.train_locations = ["Rowancreek", "Alexander"]
        self.test_location = "Covington"
        self.vali_samples = int(0.3 * num_samples)  # 30% of training samples for validation

    def load_data(self):
        """
        Load the training, validation, and test data from the specified locations and return TensorFlow datasets.
        
        :return: tuple of tf.data.Dataset objects (train_dataset, vali_dataset) if mode is 'train'
                 or tf.data.Dataset object (test_dataset) if mode is 'test'
        """
        if self.mode == 'train':
            train_data, train_label = self._load_and_sample_data(self.train_locations, "train", self.num_samples)
            vali_data, vali_label = self._load_and_sample_data(self.train_locations, "train", self.vali_samples)
            
            train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_label)).batch(self.batch_size)
            vali_dataset = tf.data.Dataset.from_tensor_slices((vali_data, vali_label)).batch(self.batch_size)
            return train_dataset, vali_dataset

        elif self.mode == 'test':
            test_data, test_label = self._load_test_data(self.test_location)
            test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_label)).batch(self.batch_size)
            return test_dataset

    def _load_and_sample_data(self, locations, data_type, num_samples):
        """
        Load and randomly sample data from the given locations.
        
        :param locations: list of str, locations to load data from
        :param data_type: str, type of data to load ('train' or 'vali')
        :param num_samples: int, number of samples to load
        :return: tuple of np.array (sampled_data, sampled_labels)
        """
        data = []
        labels = []

        for loc in locations:
            data_path = os.path.join(self.data_path, loc, f'{data_type}_data.npy')
            label_path = os.path.join(self.data_path, loc, f'{data_type}_label.npy')
            
            data_array = np.load(data_path)
            label_array = np.load(label_path)
            
            indices = random.sample(range(data_array.shape[0]), num_samples)
            
            data.append(data_array[indices])
            labels.append(label_array[indices])
        
        data = np.concatenate(data, axis=0)
        labels = np.concatenate(labels, axis=0)
        
        return data, labels

    def _load_test_data(self, location):
        """
        Load the test data from the specified location.
        
        :param location: str, location to load test data from
        :return: tuple of np.array (test_data, test_labels)
        """
        data_path = os.path.join(self.data_path, location, 'bottom_half_test_data.npy')
        label_path = os.path.join(self.data_path, location, 'bottom_half_test_label.npy')
        
        test_data = np.load(data_path)
        test_labels = np.load(label_path)
        
        return test_data, test_labels