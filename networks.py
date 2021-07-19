import os
import tempfile
import tensorflow as tf

def classifier_layers(embedding, num_classes, activation='softmax'):
    assert embedding.shape[-1] == num_classes
    
    with tf.name_scope("classifier"):
        if activation is not None:
            net = tf.keras.layers.Conv2D(num_classes, (1,1), activation=activation, padding='same', name='logits')(embedding)
        else:
            net = tf.keras.layers.Conv2D(num_classes, (1,1), padding='same', name='logits')(embedding)
        
    return net