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

# Dice coefficient with raw label masking
def dice_coefficient_w_mask(y_true, y_pred):
    """
    Compute the Dice coefficient for the filtered label, using the raw label
    to filter out pixels that are not 0 or 1.

    Args:
        y_true: Tensor with shape (..., 2), where:
                y_true[..., 0] = raw label (used for masking)
                y_true[..., 1] = filtered label (used for loss calculation)
        y_pred: Tensor of predicted labels.

    Returns:
        Dice coefficient as a float tensor.
    """
    smooth = 1e-6  # Small constant to avoid division by zero

    # Extract raw and filtered labels
    raw_y_true = tf.cast(y_true[..., 0], tf.float32)
    filtered_y_true = tf.cast(y_true[..., 1], tf.float32)

    # Flatten the tensors
    raw_y_true_f = tf.reshape(raw_y_true, [-1])
    filtered_y_true_f = tf.reshape(filtered_y_true, [-1])
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)

    # Create a mask using the raw label (include only values 0 and 1)
    mask = tf.logical_or(tf.equal(raw_y_true_f, 0), tf.equal(raw_y_true_f, 1))
    mask = tf.cast(mask, tf.float32)

    # Apply the mask to the filtered label and predicted label
    filtered_y_true_f_masked = filtered_y_true_f * mask
    y_pred_f_masked = y_pred_f * mask

    # Compute intersection and union only on masked values
    intersection = tf.reduce_sum(filtered_y_true_f_masked * y_pred_f_masked)
    denominator = tf.reduce_sum(filtered_y_true_f_masked) + tf.reduce_sum(y_pred_f_masked)

    # Avoid dividing by zero
    return (2. * intersection + smooth) / (denominator + smooth)


# Dice loss to be minimized
def dice_loss_w_mask(y_true, y_pred):
    return 1 - dice_coefficient_w_mask(y_true, y_pred)

