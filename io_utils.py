import tensorflow as tf

import numpy as np
import random

import cv2

def read_list_file(file_name):
    # Designed to read a file that contains a list of file-names as entries, one per row
    
    f = open(file_name)
    
    res = []
    for line in f:
        res.append(line.strip())
    return res


def sample_batch(data_dir, data_list, batch_size=20, seed=42):
    # Following the code at 
    # https://github.com/carrenD/Medical-Cross-Modality-Domain-Adaptation
    # https://github.com/cchen-cc/SIFA/blob/SIFA-v1/data_loader.py

    # data_dir - parent data directory
    # data_list - a list of samples to consider
    # batch_size - number of distinct images to be sampled from data_list

    random.seed(seed)
    fn = random.sample(data_list, batch_size)
    fn = [data_dir + x for x in fn]
    
    # Create a description of the features.
    feature_description = {
                'dsize_dim0': tf.io.FixedLenFeature([], tf.int64), # 256
                'dsize_dim1': tf.io.FixedLenFeature([], tf.int64), # 256
                'dsize_dim2': tf.io.FixedLenFeature([], tf.int64), # 3
                'lsize_dim0': tf.io.FixedLenFeature([], tf.int64), # 256
                'lsize_dim1': tf.io.FixedLenFeature([], tf.int64), # 256
                'lsize_dim2': tf.io.FixedLenFeature([], tf.int64), # 3
                'data_vol': tf.io.FixedLenFeature([], tf.string),  # (256*256*3, )
                'label_vol': tf.io.FixedLenFeature([], tf.string)} # (256*256*3, )

    raw_size = [256, 256, 3] # original raw input size
    volume_size = [256, 256, 3] # volume size after processing
    label_size = [256, 256, 1]

    raw_dataset = tf.data.TFRecordDataset(fn)
    
    def _parse_function(example_proto):
        # Parse the input `tf.train.Example` proto using the dictionary above.
        return tf.io.parse_single_example(example_proto, feature_description)
    parsed_dataset = raw_dataset.map(_parse_function)
    
    images = []
    labels = []
    
    for p in parsed_dataset:
        data_vol = tf.io.decode_raw(p['data_vol'], tf.float32)
        label_vol = tf.io.decode_raw(p['label_vol'], tf.float32)

        data_vol = tf.reshape(data_vol, raw_size)
        label_vol = tf.reshape(label_vol, raw_size)
        data_vol = tf.slice(data_vol, [0,0,0], volume_size)
        label_vol = tf.slice(label_vol, [0,0,1], label_size)
        
        images.append(data_vol.numpy())
        labels.append(label_vol.numpy())
    
    images = np.array(images)
    labels = np.array(labels)

    # Make sure there are no labels with a non-zero fractional part
    assert np.sum(labels == labels.astype(np.int32)) == np.prod(labels.shape)

    labels = labels.astype(int)
        
    return images,labels

# https://www.tensorflow.org/tutorials/load_data/tfrecord
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# https://stackoverflow.com/questions/33849617/how-do-i-convert-a-directory-of-jpeg-images-to-tfrecords-file-in-tensorflow/33862568#33862568
def to_tfrecord(X, Y, fn):
    # Write (256,256,3) slices and (256,256,3) labels to tfrecord.
    # Images and labels are both float32 to match the I/O from mmwhs:
    # https://github.com/carrenD/Medical-Cross-Modality-Domain-Adaptation/blob/master/source_segmenter.py line 343
    X = X.astype(np.float32)
    Y = Y.astype(np.float32)

    assert np.frombuffer(X.tobytes(), dtype=np.float32).shape[0] == 256*256*3
    assert np.frombuffer(Y.tobytes(), dtype=np.float32).shape[0] == 256*256*3
    
    with tf.io.TFRecordWriter(fn) as writer:
        example = tf.train.Example(features=tf.train.Features(feature={
            'dsize_dim0': _int64_feature(X.shape[0]),
            'dsize_dim1': _int64_feature(X.shape[1]),
            'dsize_dim2': _int64_feature(X.shape[2]),
            'lsize_dim0': _int64_feature(Y.shape[0]),
            'lsize_dim1': _int64_feature(Y.shape[1]),
            'lsize_dim2': _int64_feature(Y.shape[2]),
            'data_vol': _bytes_feature(X.tobytes()), 
            'label_vol': _bytes_feature(Y.tobytes())}))
        writer.write(example.SerializeToString())

def get_consecutive_slices(scan, labels, idx, target_shape=(256,256,3)):
    # Returns three consecutive slices from scan and labels from idx-1 to idx+1
    # If idx-1 < 0, or idx+1 > scan.shape[-1]-1 repeats the middle slice
    # Images are channel last

    # Verify images are channel last
    assert scan.shape[0] == scan.shape[1]
    assert labels.shape[0] == labels.shape[1]
    assert scan.shape == labels.shape

    X = np.zeros((scan.shape[0], scan.shape[1], 3))
    Y = np.zeros((scan.shape[0], scan.shape[1], 3), dtype=np.int32)
        
    # Compute the default image
    for channel_idx in range(3):
        i = idx + channel_idx - 1
        i = max(i, 0)
        i = min(i, scan.shape[2] - 1)
        
        X[..., channel_idx] = np.copy(scan[...,i])
        Y[..., channel_idx] = np.copy(labels[...,i])

    if X.shape != target_shape:
        X = cv2.resize(X, (256,256), interpolation=cv2.INTER_CUBIC)
        Y = cv2.resize(Y, (256,256), interpolation=cv2.INTER_NEAREST)

    return X,Y
