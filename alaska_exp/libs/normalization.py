class Normalization:
    def __init__(self, normalization_type):
        self.normalization_type = normalization_type

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


# Example usage:
# normalizer = Normalization(normalization_type='0')
# normalized_data = normalizer.normalize_data(data_to_normalize)