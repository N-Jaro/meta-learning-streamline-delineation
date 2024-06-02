import numpy as np
import tensorflow as tf
import os
import random

class MixedDataLoader:
    def __init__(self, data_path, num_samples, mode='train'):
        """
        Initialize the DataLoader with the path to data, the number of samples to load from each location,
        and the mode (train or test).
        
        :param data_path: str, path to the main data directory
        :param num_samples: int, number of samples to load from each location for training and validation
        :param mode: str, either 'train' or 'test', specifying the mode of operation
        """
        self.data_path = data_path
        self.num_samples = num_samples
        self.mode = mode
        self.train_locations = ["Rowancreek", "Alexander"]
        self.test_location = "Covington"

    def load_data(self):
        """
        Load the training, validation, and test data from the specified locations and return TensorFlow datasets.
        
        :return: tuple of tf.data.Dataset objects (train_dataset, vali_dataset, test_dataset) if mode is 'train'
                 or tf.data.Dataset object (test_dataset) if mode is 'test'
        """
        if self.mode == 'train':
            train_data, train_label = self._load_and_sample_data(self.train_locations, "train")
            vali_data, vali_label = self._load_and_sample_data(self.train_locations, "vali")
            
            train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_label))
            vali_dataset = tf.data.Dataset.from_tensor_slices((vali_data, vali_label))
            return train_dataset, vali_dataset

        elif self.mode == 'test':
            test_data, test_label = self._load_test_data(self.test_location)
            test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_label))
            return test_dataset

    def _load_and_sample_data(self, locations, data_type):
        """
        Load and randomly sample data from the given locations.
        
        :param locations: list of str, locations to load data from
        :param data_type: str, type of data to load ('train' or 'vali')
        :return: tuple of np.array (sampled_data, sampled_labels)
        """
        data = []
        labels = []

        for loc in locations:
            data_path = os.path.join(self.data_path, loc, f'{data_type}_data.npy')
            label_path = os.path.join(self.data_path, loc, f'{data_type}_label.npy')
            
            data_array = np.load(data_path)
            label_array = np.load(label_path)
            
            indices = random.sample(range(data_array.shape[0]), self.num_samples)
            
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

# # Example usage for training
# data_path = "path_to_your_data"
# num_samples = 100  # Number of samples from each location for training and validation
# data_loader = DataLoader(data_path, num_samples, mode='train')
# train_dataset, vali_dataset = data_loader.load_data()

# # Example usage for testing
# data_loader_test = DataLoader(data_path, num_samples, mode='test')
# test_dataset = data_loader_test.load_data()
