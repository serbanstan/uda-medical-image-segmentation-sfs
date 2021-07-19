# Code adapted from 
# 1. https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/voc-fcn8s/net.py
# 2. https://github.com/YangZhang4065/AdaptationSeg/blob/master/FCN_da.py

import numpy as np

import tensorflow as tf
import tensorflow.keras.backend as K

# Adapted from https://github.com/YangZhang4065/AdaptationSeg/blob/master/train_val_FCN_DA.py
# Is this the U-net loss ?
def weighted_ce_loss(num_classes = 20, class_to_ignore = None):
    mask = np.ones(num_classes)
    if class_to_ignore is not None:
        mask[class_to_ignore] = 0
    mask = K.variable(mask, dtype='float32')

    def wce_loss(y_true, y_pred, from_logits=True):
        # Preprocess data
        if from_logits == True:
            y_pred = K.softmax(y_pred, axis = -1)

        # See https://stackoverflow.com/questions/33712178/tensorflow-nan-bug
        y_pred = tf.clip_by_value(y_pred, 1e-10, 1)

        # Number of pixels in each class
        sample_num_per_class = K.sum(y_true, axis=[-3,-2],keepdims=True)

        # Indices of positive classes
        class_ind = K.cast(K.greater(sample_num_per_class, 0.), 'float32')

        # Average number of samples per class
        avg_sample_num_per_class = tf.math.divide_no_nan(K.sum(sample_num_per_class, axis=-1, keepdims=True), \
                                                        K.sum(class_ind, axis=-1, keepdims=True))
        # avg_sample_num_per_class = K.sum(sample_num_per_class, axis=-1, keepdims=True) / K.sum(class_ind, axis=-1, keepdims=True)

        # Weight of each class for each input sample
        sample_weight_per_class = tf.math.divide_no_nan(avg_sample_num_per_class, sample_num_per_class)
        # sample_weight_per_class = avg_sample_num_per_class /(sample_num_per_class+.1)

        # Compute y log y_hat for each pixel
        pixel_wise_loss = -y_true * K.log(y_pred)

        # Compute the weighted pixel loss
        weighted_pixel_wise_loss = pixel_wise_loss * sample_weight_per_class * mask

        weighted_pixel_wise_loss = K.sum(weighted_pixel_wise_loss, axis=-1, keepdims=True)

        return K.mean(weighted_pixel_wise_loss)

    return wce_loss



def masked_ce_loss(num_classes = 20, class_to_ignore = None):
    mask = np.ones(num_classes)
    if class_to_ignore is not None:
        mask[class_to_ignore] = 0
    mask = tf.keras.backend.variable(mask, dtype='float32')

    def masked_loss(y_true, y_pred, from_logits=True):
        # Preprocess data
        if from_logits == True:
            y_pred = tf.keras.backend.softmax(y_pred, axis = -1)
        
        # See https://stackoverflow.com/questions/33712178/tensorflow-nan-bug
        y_pred = tf.clip_by_value(y_pred, 1e-10, 1)

        loss = tf.keras.backend.categorical_crossentropy(y_true * mask, y_pred, axis=-1)

        return tf.keras.backend.mean(loss)

    return masked_loss
