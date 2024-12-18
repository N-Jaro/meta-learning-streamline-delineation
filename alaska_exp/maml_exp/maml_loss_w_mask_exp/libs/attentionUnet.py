from tensorflow.keras.layers import ( Input, Conv2D, BatchNormalization, Activation, Conv2DTranspose, 
                                        MaxPooling2D, Layer, add, multiply, GlobalAveragePooling2D, 
                                        Dense, Reshape, Multiply )
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

class ChannelAttention(Layer):
    def __init__(self, reduction_ratio=8, **kwargs):
        super(ChannelAttention, self).__init__(**kwargs)
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        channel = input_shape[-1]
        self.fc1 = Dense(channel // self.reduction_ratio, activation='relu', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
        self.fc2 = Dense(channel, activation='sigmoid', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
        super(ChannelAttention, self).build(input_shape)

    def call(self, inputs):
        avg_pool = GlobalAveragePooling2D()(inputs)
        avg_pool = Reshape((1, 1, avg_pool.shape[1]))(avg_pool)
        fc1_out = self.fc1(avg_pool)
        fc2_out = self.fc2(fc1_out)
        scale = Multiply()([inputs, fc2_out])
        return scale

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(ChannelAttention, self).get_config()
        config.update({'reduction_ratio': self.reduction_ratio})
        return config

class SpatialAttention(Layer):
    def __init__(self, kernel_size=7, **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.conv = Conv2D(1, self.kernel_size, padding='same', activation='sigmoid', kernel_initializer='he_normal', use_bias=False)
        super(SpatialAttention, self).build(input_shape)

    def call(self, inputs):
        avg_pool = K.mean(inputs, axis=3, keepdims=True)
        max_pool = K.max(inputs, axis=3, keepdims=True)
        concat = K.concatenate([avg_pool, max_pool], axis=3)
        attention = self.conv(concat)
        return multiply([inputs, attention])

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(SpatialAttention, self).get_config()
        config.update({'kernel_size': self.kernel_size})
        return config

class AttentionUnet:
    def __init__(self, input_shape = (224,224,8), output_mask_channels=1, filters=32, last_dropout=0.2):
        self.input_shape = input_shape
        self.output_mask_channels = output_mask_channels
        self.filters = filters
        self.last_dropout = last_dropout

    def residual_cnn_block(self, x, size, dropout=0.0, batch_norm=True):
        conv = Conv2D(size, (3, 3), padding='same')(x)
        if batch_norm:
            conv = BatchNormalization()(conv)
        conv = Activation('relu')(conv)
        conv = Conv2D(size, (3, 3), padding='same')(conv)
        if batch_norm:
            conv = BatchNormalization()(conv)
        conv = Activation('relu')(conv)
        return conv

    def attention_up_and_concatenate(self, inputs, attention_type):
        g, x = inputs
        if attention_type == 'spatial':
            attention_layer = SpatialAttention()
        elif attention_type == 'channel':
            attention_layer = ChannelAttention()
        x = attention_layer(x)
        inter_channel = x.get_shape().as_list()[3]
        g = Conv2DTranspose(inter_channel, (3,3), strides=(2, 2), padding='same')(g)
        return add([g, x])

    def build_model(self):
        inputs = Input(self.input_shape)
        filters = self.filters

        conv_224 = self.residual_cnn_block(inputs, filters)
        pool_112 = MaxPooling2D(pool_size=(2, 2))(conv_224)
        conv_112 = self.residual_cnn_block(pool_112, filters * 2)
        pool_56 = MaxPooling2D(pool_size=(2, 2))(conv_112)
        conv_56 = self.residual_cnn_block(pool_56, filters * 4)
        pool_28 = MaxPooling2D(pool_size=(2, 2))(conv_56)
        conv_28 = self.residual_cnn_block(pool_28, filters * 8)
        pool_14 = MaxPooling2D(pool_size=(2, 2))(conv_28)
        conv_14 = self.residual_cnn_block(pool_14, filters * 16)
        pool_7 = MaxPooling2D(pool_size=(2, 2))(conv_14)
        conv_7 = self.residual_cnn_block(pool_7, filters * 32)

        # Upsampling path
        up_14 = self.attention_up_and_concatenate([conv_7, conv_14], 'spatial')
        up_conv_14 = self.residual_cnn_block(up_14, filters * 16)
        up_28 = self.attention_up_and_concatenate([up_conv_14, conv_28], 'spatial')
        up_conv_28 = self.residual_cnn_block(up_28, filters * 8)
        up_56 = self.attention_up_and_concatenate([up_conv_28, conv_56], 'channel')
        up_conv_56 = self.residual_cnn_block(up_56, filters * 4)
        up_112 = self.attention_up_and_concatenate([up_conv_56, conv_112], 'channel')
        up_conv_112 = self.residual_cnn_block(up_112, filters * 2)
        up_224 = self.attention_up_and_concatenate([up_conv_112, conv_224], 'channel')
        up_conv_224 = self.residual_cnn_block(up_224, filters, dropout=self.last_dropout)

        # Output layer
        conv_final = Conv2D(self.output_mask_channels, (1, 1), activation='sigmoid')(up_conv_224)

        # Create model
        model = Model(inputs, conv_final, name="AttentionUnet")
        return model
