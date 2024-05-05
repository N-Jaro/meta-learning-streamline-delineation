import tensorflow as tf

class UNet(tf.keras.Model):
    def __init__(self, input_shape=(224, 224, 8), num_classes=1):
        super(UNet, self).__init__()

        # Downsampling Layers
        self.conv1 = self._conv_block(16, (3, 3))
        self.maxpool1 = tf.keras.layers.MaxPooling2D((2, 2))

        # Bottleneck Layers
        self.conv5 = self._conv_block(64, (3, 3))

        # Upsampling Layers
        self.upconv6 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')
        self.conv6 = self._conv_block(16, (3, 3))

        # Output Layer
        self.output_conv = tf.keras.layers.Conv2D(num_classes, (1, 1), activation='sigmoid')

    def _conv_block(self, filters, kernel_size):
        """Helper function to create a convolutional block with Conv2D, ReLU, and Dropout."""
        return tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters, kernel_size, activation='relu', kernel_initializer='he_normal', padding='same'),
            tf.keras.layers.Dropout(0.1),  # Adjust dropout as needed
            tf.keras.layers.Conv2D(filters, kernel_size, activation='relu', kernel_initializer='he_normal', padding='same')
        ])

    def call(self, inputs):
        c1 = self.conv1(inputs)
        p1 = self.maxpool1(c1)

        c5 = self.conv5(p1)

        u6 = self.upconv6(c5)
        u6 = tf.keras.layers.concatenate([u6, c1])
        c6 = self.conv6(u6)

        outputs = self.output_conv(c6)
        return outputs