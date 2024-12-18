import tensorflow as tf
from keras import backend as K

# Dice coefficient (similar to F1 score but differentiable)
def dice_coefficient(y_true, y_pred):
    smooth = 1e-6  # Small constant to avoid division by zero
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

# Dice loss to be minimized
def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)
