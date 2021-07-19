import tensorflow as tf
import numpy as np

import os
import tempfile

def vgg16_backbone(img_shape, regularizer=tf.keras.regularizers.l2(0.001), dropout=None):
    original = tf.keras.applications.VGG16(include_top=False, \
                                     weights='imagenet', \
                                     input_tensor=None, \
                                     input_shape=img_shape, \
    #                                  pooling=None, \
    #                                  classes=5, \
    #                                  classifier_activation='softmax'
                                    )

    # Add network regularizers
    for layer in original.layers:
        if regularizer is not None:
            for attr in ['kernel_regularizer', 'bias_regularizer', 'activity_regularizer']:
                if hasattr(layer, attr):
                    setattr(layer, attr, regularizer)

        if "block4_conv" in layer.name:
            layer.dilation_rate = (2,2)
        elif "block5_conv" in layer.name:
            layer.dilation_rate = (4,4)

    # After the network is changed, it needs to be reloaded for changes to take effect
    # Not sure if there is a more clean solution for this
    tmp = np.random.randint(1e10)
    tmp_weights_path = os.path.join(tempfile.gettempdir(), 'vgg16_' + str(tmp) + '_tmp_weights.h5')

    original.save_weights(tmp_weights_path)
    original = tf.keras.models.model_from_json(original.to_json())
    original.load_weights(tmp_weights_path)

    # Remove some of the layers to make the architecture compatible with dlv3
    drop_layers = ["block4_pool", "block5_pool"]

    input_layer = x = original.input
    for layer in original.layers[1:]:
        if layer.name not in drop_layers:
            x = layer(x)
        elif "block4_conv" in layer.name:
            x = layer(x, dilation_rate=(2,2))
        elif "block5_conv" in layer.name:
            x = layer(x, dilation_rate=(4,4))

        if dropout is not None:
            if "conv" in layer.name or "input" in layer.name:
                x = tf.keras.layers.Dropout(dropout)(x)

    return input_layer, x


'''
    Add the deeplabv3 decoder to an already established backcbone.

    img_shape - shape of input image
    num_classes - number of output channels
    backbone - currently 'vgg16'
    activation - whether the last layer will have an activation function applied to it
    regularizer - in the case of resnet101, regularizers may be added to the network backbone
'''
def deeplabv3(img_shape=(256,256,3), num_classes=5, backbone = 'vgg16', activation=None, regularizer=tf.keras.regularizers.l2(0.001), dropout=None):
    assert backbone in ['vgg16']

    if backbone == 'vgg16':
        input_layer, x = vgg16_backbone(img_shape, regularizer=regularizer, dropout=dropout)

    x0 = tf.keras.layers.Conv2D(256, 1, use_bias=False)(x)
    x0 = tf.keras.layers.BatchNormalization()(x0)
    x0 = tf.keras.layers.Activation('relu')(x0)

    # ASPP Conv
    dilation=12
    x1 = tf.keras.layers.ZeroPadding2D(padding=dilation)(x)
    x1 = tf.keras.layers.Conv2D(256, 3, dilation_rate=dilation, use_bias=False)(x1)
    x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.Activation('relu')(x1)

    dilation=24
    x2 = tf.keras.layers.ZeroPadding2D(padding=dilation)(x)
    x2 = tf.keras.layers.Conv2D(256, 3, dilation_rate=dilation, use_bias=False)(x2)
    x2 = tf.keras.layers.BatchNormalization()(x2)
    x2 = tf.keras.layers.Activation('relu')(x2)

    dilation=36
    x3 = tf.keras.layers.ZeroPadding2D(padding=dilation)(x)
    x3 = tf.keras.layers.Conv2D(256, 3, dilation_rate=dilation, use_bias=False)(x3)
    x3 = tf.keras.layers.BatchNormalization()(x3)
    x3 = tf.keras.layers.Activation('relu')(x3)

    # ASPP Pooling
    # size = x.shape[1:3]
    x4 = tf.keras.layers.GlobalAveragePooling2D()(x)
    x4 = tf.keras.layers.Lambda(lambda xx: tf.keras.backend.expand_dims(xx, 1))(x4)
    x4 = tf.keras.layers.Lambda(lambda xx: tf.keras.backend.expand_dims(xx, 1))(x4)
    x4 = tf.keras.layers.Conv2D(256, 1, use_bias=False)(x4)
    x4 = tf.keras.layers.BatchNormalization()(x4)
    x4 = tf.keras.layers.Activation('relu')(x4)

    x4 = tf.keras.layers.Lambda(lambda xx, target_shape: tf.compat.v1.image.resize(xx,
                                                           target_shape,
                                                           method='bilinear', 
                                                           align_corners=False),
                                                           arguments={'target_shape':x.shape[1:3]},
                                                           # arguments={'target_shape':(64,128)},
                                                           name='pooling_resizing_layer')(x4) 


    x = tf.keras.layers.Concatenate()([x0, x1, x2, x3, x4])

    # Project
    x = tf.keras.layers.Conv2D(256, 1, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(.5)(x)

    # Post Projection
    x = tf.keras.layers.ZeroPadding2D(1)(x)
    x = tf.keras.layers.Conv2D(256, 3, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(num_classes, 1)(x)

    # Final resizing
    x = tf.keras.layers.Lambda(lambda xx, target_shape: tf.compat.v1.image.resize(xx,
                                                           target_shape,
                                                           method='bilinear', 
                                                           align_corners=False),
                                                           arguments={'target_shape':img_shape[:2]},
                                                           # arguments={'target_shape':(512, 1024)},
                                                           name='final_resizing_layer')(x)

    if activation is not None:
        x = tf.keras.layers.Activation(activation)(x)

    model = tf.keras.Model(input_layer, x)

    return model


