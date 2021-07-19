# Contains utilities needed to process images/labels from image segmentation datasets

import tensorflow as tf
import numpy as np

import os
import io_utils as io

from PIL import Image
import cv2
import SimpleITK as sitk
import scipy.ndimage as ndimage

# Label ids for the mmwhs dataset
# From https://github.com/carrenD/Medical-Cross-Modality-Domain-Adaptation/blob/master/source_segmenter.py
label_ids_mmwhs = {"ignore": 0,
    "lv_myo": 1,
    "la_blood": 2,
    "lv_blood": 3,
    "aa": 4,
}

# we use the following final mapping for labels
label_ids_abdomen = {"ignore": 0,
    "liver": 1,
    "right_kidney": 2,
    "left_kidney": 3,
    "spleen": 4,
}

def normalize(X):
	ans = np.copy(X)

	ans = ans - np.mean(ans)
	ans = ans / np.std(ans)

	thresh = 3
	ans[ans > thresh] = thresh
	ans[ans < -thresh] = -thresh

	return ans


# Preprocess abdominal CT scans
def preprocess_abdomen_ct(raw_ct_img_dir, imf, labelf, final_cropping=True):
    # CT specific processing - code from
    # https://github.com/assassint2017/abdominal-multi-organ-segmentation/blob/master/data_perpare/get_data.py
    upper = 275
    lower = -125
    down_scale = 0.5
    expand_slice = 10
    
    ct = sitk.ReadImage(raw_ct_img_dir + "img/" + imf, sitk.sitkInt16)
    ct_array = sitk.GetArrayFromImage(ct)

    seg = sitk.ReadImage(raw_ct_img_dir + "label/" + labelf, sitk.sitkInt8)
    seg_array = sitk.GetArrayFromImage(seg)
    
    print("original shape", ct_array.shape)
    ct_array = ndimage.zoom(ct_array, (1, down_scale, down_scale), order=3)
    seg_array = ndimage.zoom(seg_array, (1, down_scale, down_scale), order=0)

    ct_array[ct_array > upper] = upper
    ct_array[ct_array < lower] = lower
    
    ############################################################################
    # Older pre-processing
    scan = np.copy(ct_array)
    labels = np.copy(seg_array)
    
    # Keep only four classes
    adjusted_labels = np.zeros(labels.shape, dtype=np.int32)
    adjusted_labels[labels == 6] = 1
    adjusted_labels[labels == 2] = 2
    adjusted_labels[labels == 3] = 3
    adjusted_labels[labels == 1] = 4
    labels = np.copy(adjusted_labels)
    
    # Make the immages channel last
    scan = np.moveaxis(scan, 0, -1)
    labels = np.moveaxis(labels, 0, -1)
    
    # Adjust scan alignment
    scan = np.flip(scan, 0)
    labels = np.flip(labels, 0)
    
    # Normalize the image
    scan = normalize(scan)

    if final_cropping:
        # Remove 0 label space around the image
        # From a 256x256xC scan, space is removed 30 units up, down, left or right of the furthest labeled pixel
        # Then, the image is resized back to 256x256
        
        imin = 1000
        imax = 0
        jmin = 1000
        jmax = 0
        for c in range(scan.shape[-1]):
            for i in range(scan.shape[0]):
                for j in range(scan.shape[1]):
                    if labels[i,j,c] != 0:
                        imin = min(imin, i)
                        imax = max(imax, i)
                        jmin = min(jmin, j)
                        jmax = max(jmax, j)
        
        # Add extra buffer around the labeled regions
        disp = 30
        imin = max(imin - disp, 0)
        imax = min(imax + disp, scan.shape[0])
        jmin = max(jmin - disp, 0)
        jmax = min(jmax + disp, scan.shape[1])
        
        for c in range(scan.shape[-1]):
            for i in range(scan.shape[0]):
                for j in range(scan.shape[1]):
                    if imin <= i and i <= imax and jmin <= j and j <= jmax:
                        continue
                    
                    assert labels[i,j,c] == 0
        
        # Remove some extra space on the border of the images
        for c in range(scan.shape[-1]):
            scan[...,c] = cv2.resize(scan[...,c][imin:imax,jmin:jmax], (256,256), interpolation=cv2.INTER_CUBIC)
            labels[...,c] = cv2.resize(labels[...,c][imin:imax,jmin:jmax], (256,256), interpolation=cv2.INTER_NEAREST)

    print("final shape", scan.shape)
    
    return scan,labels