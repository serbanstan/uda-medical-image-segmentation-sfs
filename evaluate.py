import numpy as np

import medpy.metric.binary as mmb

import io_utils as io
import data_utils

import os

def evaluate_mmwhs(data_dir, data_list, model, label_ids, id_to_ignore = 0):
    # Evaluation function
    # Instead of re-training the model and reporting mean/std based on different random seeds, instead we just
    # report the mean/std based on the four test images. This seems to be the method used in 
    # https://github.com/cchen-cc/SIFA/blob/master/evaluate.py
    # Thus, for each test image we obtain prediction labels, and find DICE and ASSM results
    # We make use of the pre-processing we did. Test tf-records 0..255 will correspond to image1, 
    # test tf-records 256..511 will correspond to image2 etc. 
    
    # Initialize result dicts
    dice = {}
    assd = {}
    for label in label_ids:
        if label_ids[label] == id_to_ignore:
            continue
            
        dice[label] = []
        assd[label] = []
    
    # Find results for each class
    for test_image in range(4):
        # The 3D labels and our predictions
        y_true = []
        y_hat = []

        for i in range(test_image * 256, (test_image+1) * 256):
            # We read in the i'th slice and compute predictions
            X_slice,y_true_slice = io.sample_batch(data_dir, [data_list[i]], 1)

            y_true.append(np.copy(y_true_slice))
            y_hat.append(model.predict(X_slice))
        
        y_true = np.array(y_true).reshape(256,256,256)
        y_hat = np.array(y_hat).reshape(256,256,256)
        
        print(y_true.shape, y_hat.shape)
            
        for label in label_ids:
            if label_ids[label] == id_to_ignore:
                continue
            
            # Prep data and compute metrics
            curr_y_true = np.copy(y_true)
            curr_y_true[curr_y_true != label_ids[label]] = 0
            
            curr_y_hat = np.copy(y_hat)
            curr_y_hat[curr_y_hat != label_ids[label]] = 0
            
            dice[label].append(mmb.dc(curr_y_hat, curr_y_true))
            assd[label].append(mmb.assd(curr_y_hat, curr_y_true))
    
    # Compute mean/std for each class for dice and assd
    dice_mu = {}
    dice_sd = {}
    dice_total_mu = []
    
    assd_mu = {}
    assd_sd = {}
    assd_total_mu = []
    
    for label in label_ids:
        if label_ids[label] == id_to_ignore:
            continue
            
        dice_mu[label] = np.mean(dice[label])
        dice_sd[label] = np.std(dice[label])
        dice_total_mu.append(dice_mu[label])
        
        assd_mu[label] = np.mean(assd[label])
        assd_sd[label] = np.std(assd[label])
        assd_total_mu.append(assd_mu[label])
        
    dice_total_mu = np.mean(dice_total_mu)
    assd_total_mu = np.mean(assd_total_mu)
    
    return dice_total_mu, dice_mu, dice_sd, assd_total_mu, assd_mu, assd_sd


def evaluate_abdomen(raw_ct_img_dir, data_seed, model, label_ids, id_to_ignore = 0, whole_dataset_computation=False):
    # Use the same random seed as in the data processing notebook
    assert data_seed == 0
    np.random.seed(data_seed)
    train_indices = sorted(np.random.choice(range(30), 24, replace=False))
    test_indices = np.asarray(sorted([x for x in range(30) if x not in train_indices]))
    test_images = np.asarray(sorted(os.listdir(raw_ct_img_dir + "img/")))[test_indices]
    test_labels = np.asarray(sorted(os.listdir(raw_ct_img_dir + "label/")))[test_indices]

    print(test_images)

    all_scans = []
    all_segmaps = []
    for imf,labelf in zip(test_images, test_labels):
        print("Working on a new scan", imf, labelf)

        scan,labels = data_utils.preprocess_abdomen_ct(raw_ct_img_dir, imf, labelf, final_cropping=True)

        all_scans.append(np.copy(scan))
        all_segmaps.append(np.copy(labels))

    # Compute results for all images at once
    if whole_dataset_computation == True:
        combined_scans = []
        for scan in all_scans:
            combined_scans.append(np.moveaxis(np.copy(scan), 2, 0))
        combined_scans = np.vstack(combined_scans)
        combined_scans = np.moveaxis(combined_scans, 0, 2)

        combined_segmaps = []
        for segmap in all_segmaps:
            combined_segmaps.append(np.moveaxis(np.copy(segmap), 2, 0))
        combined_segmaps = np.vstack(combined_segmaps)
        combined_segmaps = np.moveaxis(combined_segmaps, 0, 2)

        all_scans = [combined_scans]
        all_segmaps = [combined_segmaps]
        
    # Initialize result dicts
    dice = {}
    assd = {}
    for label in label_ids:
        if label_ids[label] == id_to_ignore:
            continue

        dice[label] = []
        assd[label] = []

    # Find results for each class
    for scan,labels in zip(all_scans, all_segmaps):
        # The 3D labels and our predictions
        y_true = []
        y_hat = []

        for idx in range(scan.shape[-1]):
            # Reshape the images/labels to 256x256xNum_Slices
            X,_ = io.get_consecutive_slices(scan, labels, idx, target_shape=(256,256,3))

            # Ignore slices that have none of the target organs present
            if len(np.unique(labels[...,idx])) == 1:
                continue
            
            Yhat = model.predict(X.reshape(1,256,256,3))[0]
            
            y_true.append(np.copy(labels[...,idx].reshape(1,256,256)))
            y_hat.append(np.copy(Yhat.reshape(1,256,256)))
            
        y_true = np.vstack(y_true)
        y_hat = np.vstack(y_hat)

        print(y_true.shape, y_hat.shape)

        for label in label_ids:
            if label_ids[label] == id_to_ignore:
                continue

            # Prep data and compute metrics
            curr_y_true = np.copy(y_true)
            curr_y_true[curr_y_true != label_ids[label]] = 0

            curr_y_hat = np.copy(y_hat)
            curr_y_hat[curr_y_hat != label_ids[label]] = 0

            dice[label].append(mmb.dc(curr_y_hat, curr_y_true))
            assd[label].append(mmb.assd(curr_y_hat, curr_y_true))

    # Compute mean/std for each class for dice and assd
    dice_mu = {}
    dice_sd = {}
    dice_total_mu = []

    assd_mu = {}
    assd_sd = {}
    assd_total_mu = []

    for label in label_ids:
        if label_ids[label] == id_to_ignore:
            continue

        dice_mu[label] = np.mean(dice[label])
        dice_sd[label] = np.std(dice[label])
        dice_total_mu.append(dice_mu[label])

        assd_mu[label] = np.mean(assd[label])
        assd_sd[label] = np.std(assd[label])
        assd_total_mu.append(assd_mu[label])

    dice_total_mu = np.mean(dice_total_mu)
    assd_total_mu = np.mean(assd_total_mu)

    return dice_total_mu, dice_mu, dice_sd, assd_total_mu, assd_mu, assd_sd