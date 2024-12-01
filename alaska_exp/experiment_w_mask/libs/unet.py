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

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate

class DeeperUnet:
    def __init__(self, input_shape=(224, 224, 8), num_classes=1):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build_model(self):
        inputs = Input(shape=self.input_shape)
        
        # Downsampling path
        c1 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        p1 = MaxPooling2D((2, 2))(c1)
        
        c2 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(p1)
        p2 = MaxPooling2D((2, 2))(c2)
        
        c3 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(p2)
        p3 = MaxPooling2D((2, 2))(c3)
        
        # Bottleneck
        c4 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(p3)
        
        # Upsampling path
        u1 = Conv2DTranspose(128, (3, 3), strides=(2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(c4)
        u1 = concatenate([u1, c3])
        
        u2 = Conv2DTranspose(64, (3, 3), strides=(2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(u1)
        u2 = concatenate([u2, c2])
        
        u3 = Conv2DTranspose(32, (3, 3), strides=(2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(u2)
        u3 = concatenate([u3, c1])
        
        # Output layer
        outputs = Conv2D(self.num_classes, (1, 1), activation='sigmoid', padding='same')(u3)
        
        # Create the model
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model

class SimpleAttentionUNet:
    def __init__(self, input_shape=(224, 224, 8), num_classes=1):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def attention_gate(self, g, x, filters):
        # g is the gating signal, x is the skip connection
        theta_x = tf.keras.layers.Conv2D(filters, (1, 1), strides=(1, 1), padding='same')(x)
        phi_g = tf.keras.layers.Conv2D(filters, (1, 1), strides=(1, 1), padding='same')(g)
        add_xg = tf.keras.layers.add([theta_x, phi_g])
        relu_xg = tf.keras.layers.Activation('relu')(add_xg)
        psi = tf.keras.layers.Conv2D(1, (1, 1), padding='same')(relu_xg)
        sigmoid_xg = tf.keras.layers.Activation('sigmoid')(psi)
        attention = tf.keras.layers.multiply([x, sigmoid_xg])
        return attention

    def build_model(self):
        inputs = tf.keras.Input(shape=self.input_shape)

        # Downsample
        c1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
        c1 = tf.keras.layers.Dropout(0.1)(c1)
        c1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
        p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

        c2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
        c2 = tf.keras.layers.Dropout(0.1)(c2)
        c2 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
        p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

        c3 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
        c3 = tf.keras.layers.Dropout(0.2)(c3)
        c3 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
        p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)

        c4 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
        c4 = tf.keras.layers.Dropout(0.2)(c4)
        c4 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
        p4 = tf.keras.layers.MaxPooling2D((2, 2))(c4)

        # Bottleneck
        c5 = tf.keras.layers.Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
        c5 = tf.keras.layers.Dropout(0.3)(c5)
        c5 = tf.keras.layers.Conv2D(1024, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

        # Upsample
        u6 = tf.keras.layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
        att6 = self.attention_gate(u6, c4, 512)
        u6 = tf.keras.layers.concatenate([u6, att6])
        c6 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
        c6 = tf.keras.layers.Dropout(0.2)(c6)
        c6 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

        u7 = tf.keras.layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
        att7 = self.attention_gate(u7, c3, 256)
        u7 = tf.keras.layers.concatenate([u7, att7])
        c7 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
        c7 = tf.keras.layers.Dropout(0.2)(c7)
        c7 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

        u8 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
        att8 = self.attention_gate(u8, c2, 128)
        u8 = tf.keras.layers.concatenate([u8, att8])
        c8 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
        c8 = tf.keras.layers.Dropout(0.1)(c8)
        c8 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

        u9 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
        att9 = self.attention_gate(u9, c1, 64)
        u9 = tf.keras.layers.concatenate([u9, att9])
        c9 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
        c9 = tf.keras.layers.Dropout(0.1)(c9)
        c9 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

        # Output layer
        outputs = tf.keras.layers.Conv2D(self.num_classes, (1, 1), activation='sigmoid')(c9)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model
