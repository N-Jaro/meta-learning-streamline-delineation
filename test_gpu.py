import tensorflow as tf

# Check if TensorFlow can access the GPU
if tf.config.list_physical_devices('GPU'):
    print("TensorFlow is using the GPU")
else:
    print("TensorFlow is NOT using the GPU")
