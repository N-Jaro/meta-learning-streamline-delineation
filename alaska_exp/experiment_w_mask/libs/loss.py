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


import tensorflow as tf

# Dice coefficient (similar to F1 score but differentiable)
def dice_coefficient_w_mask(y_true, y_pred):
    smooth = 1e-6  # Small constant to avoid division by zero

    # Flatten the tensors
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)

    # Create a mask to include only values 0 and 1
    mask = tf.logical_or(tf.equal(y_true_f, 0), tf.equal(y_true_f, 1))
    mask = tf.cast(mask, tf.float32)

    # Apply the mask
    y_true_f_masked = y_true_f * mask
    y_pred_f_masked = y_pred_f * mask

    # Compute intersection and union only on masked values
    intersection = tf.reduce_sum(y_true_f_masked * y_pred_f_masked)
    denominator = tf.reduce_sum(y_true_f_masked) + tf.reduce_sum(y_pred_f_masked)
    
    # Avoid dividing by zero
    return (2. * intersection + smooth) / (denominator + smooth)

# Dice loss to be minimized
def dice_loss_w_mask(y_true, y_pred):
    return 1 - dice_coefficient_w_mask(y_true, y_pred)

