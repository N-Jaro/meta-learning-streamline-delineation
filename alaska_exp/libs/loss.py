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


# # Use dice coefficient function as the loss function 
# def dice_coef(y_true, y_pred):
#     y_true_f = K.flatten(y_true)
#     y_pred_f = K.flatten(y_pred)
#     intersection = K.sum(y_true_f * y_pred_f)
#     return (2.0 * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)

# # Jacard coefficient
# def jacard_coef(y_true, y_pred):
#     y_true_f = K.flatten(y_true)
#     y_pred_f = K.flatten(y_pred)
#     intersection = K.sum(y_true_f * y_pred_f)
#     return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)

# # calculate loss value
# def jacard_coef_loss(y_true, y_pred):
#     return -jacard_coef(y_true, y_pred)

# # calculate loss value
# def dice_coef_loss(y_true, y_pred):
#     return -dice_coef(y_true, y_pred)