import tensorflow as tf

class SimpleUNet:
    def __init__(self, input_shape=(224, 224, 8), num_classes=1):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build_model(self):
        inputs = tf.keras.Input(shape=self.input_shape)

        # Downsample
        c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
        c1 = tf.keras.layers.Dropout(0.1)(c1)
        c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
        p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

        # Bottleneck
        c5 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
        c5 = tf.keras.layers.Dropout(0.2)(c5)
        c5 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

        # Upsample
        u6 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c5)
        u6 = tf.keras.layers.concatenate([u6, c1])
        c6 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
        c6 = tf.keras.layers.Dropout(0.1)(c6)
        c6 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

        # Output layer
        outputs = tf.keras.layers.Conv2D(self.num_classes, (1, 1), activation='sigmoid')(c6)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model