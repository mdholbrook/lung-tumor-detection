import os
import sys
from glob2 import glob
import pandas as pd
import numpy as np
from sklearn.metrics import auc, accuracy_score
from skimage.measure import regionprops
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from threading import Thread
from time import time

sns.set_theme(style='whitegrid', font_scale=1.1, palette='bright')


import SimpleITK as sitk
from tqdm import tqdm
from skimage.measure import label
from skimage.filters import gaussian
# plt.rcParams["figure.figsize"]=14,12

sys.path.append('../Analyis')
sys.path.append('../MachineLearning')
from MachineLearning.evaluate_results_201020 import load_evaluation_data
from MachineLearning.overlay_ims import overlay_image


def load_data(path):
    im, lab, prob, mask = load_evaluation_data(path, label_name='label.nii*')
    im = im[0].squeeze()
    lab = lab[0].squeeze()
    prob = prob[0].squeeze()
    mask = mask[0].squeeze().astype(np.uint8)
    return im, lab, prob, mask


def display_montage(im, label):
    colormap = ['black', 'red']
    sz = im.shape
    z_inds = np.linspace(0, sz[-1], 8).astype(int)[1:-1]

    im_out = np.zeros((sz[0], sz[1]*len(z_inds), 3), dtype=np.uint8)

    for i, z in enumerate(z_inds):
        im_tmp = im[:, :, z]
        lab_tmp = label[:, :, z]

        im_tmp = overlay_image(im_tmp, lab_tmp, colormap, alpha=0.5)

        im_out[:, sz[1]*i:sz[1]*(i+1),:] = im_tmp

    plt.imshow(im_out)


def load_filenames(base_path):
    # Get filenames - Tumor
    im_files = os.path.join(base_path, '*/image.nii')
    im_files = glob(im_files)

    for i in range(len(im_files)):
        im_files[i] = os.path.split(im_files[i])[0]

    return im_files


def show_images(im_file):

    im, lab, prob, mask = load_data(im_file)

    # Show prediction
    print('Labels')
    display_montage(im, lab)
    display_montage(im.T, lab.T)

    # Show prediction
    print('Prediction')
    prob_ = prob #* mask
    display_montage(im, prob_)
    display_montage(im.T, prob_.T)


def dilate_mask(mask):
    # Morphological radii
    dilate = [28]*3
    erode = [28]*3

    # Convert image to sitk
    mask_ = sitk.GetImageFromArray(mask)

    # Pad mask before dilation to avoid edge effects
    mask_ = sitk.ConstantPad(mask_, dilate, dilate, 0.0)

    # Dilate the mask
    mask_ = sitk.BinaryDilate(mask_, dilate)

    # Erode the mask
    mask_ = sitk.BinaryErode(mask_, erode)

    # Crop the mask
    mask_ = sitk.Crop(mask_, dilate, dilate)

    return sitk.GetArrayFromImage(mask_)


def erode_label(label):
    x_ = sitk.GetImageFromArray(label)
    erode = 4
    erode = [erode]*3
    # x_ = sitk.BinaryErode(x_, erode)

    return x_


def compute_regions(binary_mask):
    binary_mask = erode_label(np.uint8(binary_mask))
    binary_mask = sitk.ConnectedComponent(binary_mask)
    binary_mask = sitk.GetArrayFromImage(binary_mask)
    return binary_mask


def detection(label_regions, prediction_regions):

    label_props = regionprops(label_regions)
    pred_props = regionprops(prediction_regions)

    # Remove small regions
    min_vol = 600
    label_props = [i for i in label_props if i.area > min_vol]
    pred_props = [i for i in pred_props if i.area > min_vol]

    # Initialize outputs
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    unions_all = []
    pred_vals = [i.label for i in pred_props]

    for label in label_props:

        unions = []
        for pred in pred_props:

            # Compute direct hits
            lab = np.zeros_like(label_regions)
            prd = np.zeros_like(lab)
            lab[label.slice] = label.image
            prd[pred.slice] = pred.image

            union = np.sum(lab * prd)

            if union > 1:
                unions.append(pred.label)
                unions_all.append(pred.label)

            else:
                # Compute "near" hits
                # Compute distance between regions centers in voxels
                dist_factor = 0.75
                dist = np.sqrt(np.sum(np.square(np.array(label.centroid) - np.array(pred.centroid))))

                if dist < (dist_factor + 0.5) * label.equivalent_diameter:
                    unions.append(pred.label)
                    unions_all.append(pred.label)

        # Assign positives
        if len(unions) > 0:
            true_positives += 1
        else:
            false_negatives += 1

    # Assign all other found predictions as false positives
    false_positives = len(np.setdiff1d(pred_vals, unions_all))

    return true_positives, false_positives, false_negatives


def compute_prediction_vectors(lab, pred, lung_mask, thresholds):
    # Initialize outputs
    detection, tumor_number, voxel_numbers = [], [], []

    # Compute label region map
    lab = compute_regions(np.uint8(lab))
    label_props = regionprops(lab)

    # Remove small regions
    min_vol = 600
    label_props = [i for i in label_props if i.area > min_vol]

    # Dilate lung mask
    lung_mask = dilate_mask(lung_mask)

    # Apply lung mask
    pred = lung_mask * pred

    for t in tqdm(thresholds):

        # Threshold prediction image and filter
        thresh_pred = pred > t
        thresh_pred = gaussian(thresh_pred, sigma=1.5) > 0.5

        # Compute continuous segments
        thresh_pred = compute_regions(thresh_pred)
        pred_props = regionprops(thresh_pred)
        pred_props = [i for i in pred_props if i.area > min_vol]

        # Recompose predictions without small regions
        pred_filtered = np.zeros_like(pred)
        for p in pred_props:
            pred_filtered[p.slice] = p.image

        for i, label_region in enumerate(label_props):

            # Compute union
            union = np.sum(label_region.image * pred_filtered[label_region.slice])

            detected = False
            if union > 1:
                detected = True
            else:
                # Compute "near" hits
                # Compute distance between regions centers in voxels
                dist_factor = 0.75
                for p in pred_props:
                    dist = np.sqrt(np.sum(np.square(np.array(label_region.centroid) - np.array(p.centroid))))

                    if dist < (dist_factor + 0.5) * label_region.equivalent_diameter:
                        detected = True
                        break

            # Append to output
            tumor_number.append(label_region.label)
            detection.append(detected)
            voxel_numbers.append(label_region.area)

    return detection, tumor_number, voxel_numbers


def calcuate_features(base_path, feature_name='data.json'):

    # Output file
    base_path = os.path.join(base_path, '**')
    sname = os.path.join(os.path.split(base_path)[0], 'Analysis', feature_name)

    if os.path.exists(sname):
        df = pd.read_json(sname)

    else:
        # Load data
        im_files = load_filenames(base_path)
        thresholds = np.linspace(0.0, 1.0, 22)[1:-1]

        results = {'image_file': [], 'threshold': [], 'tumor_number': [], 'detection': [], 'num_voxels': []}
        for im_file in im_files:
            # Load images
            im, lab, pred, mask = load_data(im_file)

            # Compute true and false positives
            detections, tumor_numbers, num_voxels = compute_prediction_vectors(lab, pred, mask, thresholds)

            results['image_file'].extend([im_file]*len(detections))
            results['threshold'].extend(np.repeat(thresholds, len(np.unique(tumor_numbers))))
            results['tumor_number'].extend(tumor_numbers)
            results['detection'].extend(detections)
            results['num_voxels'].extend(num_voxels)


        # Create pandas dataframe and save
        df = pd.DataFrame.from_dict(results)
        df.to_json(sname)

    return df


def tumor_size_detection(detection_file, threshold_file):

    outpath = os.path.split(detection_file)[0]

    # Load detection file
    df = pd.read_json(detection_file)
    dx = 0.063
    vox_to_vol = lambda x: x*dx**3

    # Load thresholds
    with open(train_thresh_file, 'r') as f:
        thresholds = f.readlines()
    thresholds = [float(i) for i in thresholds]

    # Convert voxels to volume
    df['num_voxels'] = df['num_voxels'].map(vox_to_vol)

    im_files = df['image_file'].unique().tolist()
    split_base = lambda x: os.path.split(x)[0]
    im_dirs = pd.Series(im_files).map(split_base).unique().tolist()

    # bins = [0, 4000, 8000, 12000, 20000]
    bins = [0.15, 0.25, 0.5, 1.0, 5.0, 10.0]
    df_out = {'Bin': [], 'Percent': [], 'Tumors_in_bin': []}
    for i, sub in enumerate(im_dirs):
        # Separate by threshold
        df_thresh = df.loc[np.abs(df['threshold'] - thresholds[i]) < 0.01]

        # Separate by network (only one valid per threshold)
        df_sub = df_thresh.loc[df_thresh['image_file'].str.contains(sub)]
        df_sub = df_sub.filter(['detection', 'num_voxels'])

        for b in range(len(bins)-1):
            dets = df_sub.loc[df_sub['num_voxels'].between(bins[b], bins[b+1]), 'detection']
            pct = 100*dets.sum()/len(dets)
            if not np.isnan(pct):
                df_out['Bin'].append('{:0.2f} to {:0.2f}'.format(bins[b], bins[b+1]))
                # df_out['Bin'].append(bins[b+1])
                df_out['Percent'].append(pct)
                df_out['Tumors_in_bin'].append(len(dets))

    plt.close('all')
    # sns.set_palette('bright')
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.barplot(x='Bin', y='Percent', data=df_out, color='tab:blue', ci=None, ax=ax)
    plt.xlabel('Tumor Volume [mm$^3$]')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=50)
    plt.ylabel('Portion of tumors detected [%]')
    plt.ylim([0, 102])
    fig.savefig(os.path.join(outpath, 'detected_percent.png'), bbox_inches='tight', pad_inches=0.2)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.barplot(x='Bin', y='Tumors_in_bin', data=df_out, color='tab:blue', ci=None)
    plt.xlabel('Tumor Volume [mm$^3$]')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=50)
    plt.ylabel('Number of tumors')
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    fig.savefig(os.path.join(outpath, 'number of tumors.png'), bbox_inches='tight', pad_inches=0.2)


def k_fold_analysis(base_path, feature_name='data.json'):

    # Set up output file
    outpath = os.path.join(base_path, 'Analysis', os.path.splitext(feature_name)[0])
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    # Get data files
    res_files = sorted(glob(os.path.join(base_path, '**', feature_name)))

    # Plotting params
    lw = 3

    i = 0
    aucs = []
    tprs = []
    precisions = []
    recalls = []
    ideal_threshs = []
    mean_fpr = np.linspace(0, 1, 100)
    for res_file in res_files:

        # Load data
        df = pd.read_json(res_file)

        # Get thresholds
        thresholds = np.array(df['thresholds'][0])
        num_threshs = len(thresholds)

        # Get metrics
        tp = np.zeros(num_threshs)
        fn = np.zeros(num_threshs)
        fp = np.zeros(num_threshs)
        for df_ in df.iterrows():
            tp += np.array(df_[1]['true_positives'])
            fn += np.array(df_[1]['false_negatives'])
            fp += np.array(df_[1]['false_positives'])

        # Compute all positives
        all_pos = np.mean(tp + fn)

        # Add 0 and 100 to the data
        thresholds = np.append(np.insert(thresholds, 0, 0), 1.0)
        tp = np.append(np.insert(tp, 0, all_pos), 0)
        fn = np.append(np.insert(fn, 0, 0), all_pos)
        fp = np.append(np.insert(fp, 0, all_pos+np.max(fp)), 0)

        # Make false postitive rate decrease
        inds = np.diff(fp) > 0
        while any(inds):
            inds = np.where(inds)[0] + 1
            fp[inds] = fp[inds-1] #np.mean([fp[inds-1], fp[inds]], axis=0)
            inds = np.diff(fp) > 0

        # Compute rates
        tpr = tp/(all_pos)
        fpr = fp/(np.max(fp))

        # Reverse arrays
        tpr = tpr[::-1]
        fpr = fpr[::-1]

        # Compute precision and recall
        precision = tp / (tp + fp + 1e-12)
        precision[0], precision[-1] = 0, 1
        recall = tp / (tp + fn + 1e-12)
        recall[0], recall[-1] = 1, 0

        # Compute ROC and AUC
        tprs.append(np.interp(mean_fpr, xp=fpr, fp=tpr))
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

        # Compute ideal threshold
        ideal_thresh = recall - precision
        ind_min = np.argmin(np.abs(ideal_thresh))
        ind = ideal_thresh == ideal_thresh[ind_min]
        ideal_thresh = thresholds[ind].mean()

        # Save precision, recall, and ideal threshold
        precisions.append(precision)
        recalls.append(recall)
        ideal_threshs.append(ideal_thresh)

    plt.close('all')
    sns.set_theme(style='ticks', font_scale=1.1, palette='bright')
    fig = plt.figure()

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k',
             label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[0] = 0.0
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")

    fig.savefig(os.path.join(outpath, 'ROC_kfolds.png'), bbox_inches='tight', pad_inches=0.2)


    # Precision recall curves
    sns.set_theme(style='whitegrid', font_scale=1.1, palette='bright')
    fig = plt.figure()
    mean_prec = np.mean(precisions, axis=0)
    mean_reca = np.mean(recalls, axis=0)
    plt.plot(thresholds, mean_prec, color='b',
             label=r'Mean Precision',
             lw=2, alpha=.8)
    plt.plot(thresholds, mean_reca, color='r',
             label=r'Mean Recall',
             lw=2, alpha=.8)

    std_prec = np.std(precisions, axis=0)
    std_reca = np.std(recalls, axis=0)
    prec_upper = np.minimum(mean_prec + std_prec, 1)
    prec_lower = np.maximum(mean_prec - std_prec, 0)
    reca_upper = np.minimum(mean_reca + std_reca, 1)
    reca_lower = np.maximum(mean_reca - std_reca, 0)
    plt.fill_between(thresholds, prec_lower, prec_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')
    plt.fill_between(thresholds, reca_lower, reca_upper, color='grey', alpha=.2,
                     )

    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('Decision Threshold')
    plt.ylabel('')
    plt.legend(loc="lower center")

    fig.savefig(os.path.join(outpath, 'Prec_recall_kfolds.png'), bbox_inches='tight', pad_inches=0.2)

    # Save thresholds
    out_file = os.path.join(outpath, 'ideal_thresholds.txt')
    with open(out_file, 'w') as f:
        for t in ideal_threshs:
            f.writelines('{:0.3f}\n'.format(t))


def compute_total_metrics_at_threshold(test_directory, test_fname, train_thresh_file):

    # Load thresholds
    with open(train_thresh_file, 'r') as f:
        thresholds = f.readlines()
    thresholds = [float(i) for i in thresholds]

    # Get test directories
    test_files = sorted(glob(os.path.join(test_directory, '**/', test_fname)))

    fneg, fpos, tpos = [], [], []
    tfneg, tfpos, ttpos = [], [], []
    test_threshs = []
    for test_file, thresh in zip(test_files, thresholds):
        # Load detection data
        df = pd.read_json(test_file)
        num_sets = df.shape[-1]

        # Load test set data
        file = os.path.join(os.path.split(test_file)[0], 'Analysis/stats.txt')
        with open(file, 'r') as f:
            test = f.readlines()
        ntumors = int(test[1].split()[-1])
        tthresh = float(test[3].split()[-1])
        test_threshs.append(tthresh)

        # Find nearest threshold
        thresh_ind = np.argmin(np.abs(np.array(df.loc[0, 'thresholds']) - thresh))
        test_thresh_ind = np.argmin(np.abs(np.array(df.loc[0, 'thresholds']) - tthresh))

        # Get false positive rate, false negative rate, true positive rates
        for i in range(num_sets):
            fneg.append(df.loc[i, 'false_negatives'][thresh_ind])
            fpos.append(df.loc[i, 'false_positives'][thresh_ind])
            tpos.append(df.loc[i, 'true_positives'][thresh_ind])

            tfneg.append(df.loc[i, 'false_negatives'][test_thresh_ind])
            tfpos.append(df.loc[i, 'false_positives'][test_thresh_ind])
            ttpos.append(df.loc[i, 'true_positives'][test_thresh_ind])


    # Compute outputs
    tpos = np.array(tpos)
    fneg = np.array(fneg)
    fpos = np.array(fpos)
    ttpos = np.array(ttpos)
    tfneg = np.array(tfneg)
    tfpos = np.array(tfpos)

    # Compute all true positives
    tpos_all = tpos.sum()
    ttpos_all = ttpos.sum()

    # Compute precision recall
    precision = tpos.sum()/(tpos.sum() + fpos.sum())
    recall = tpos.sum()/(tpos.sum() + fneg.sum())
    f1 = 2/(1/precision + 1/recall)
    tprecision = ttpos.sum()/(ttpos.sum() + tfpos.sum())
    trecall = ttpos.sum()/(ttpos.sum() + tfneg.sum())
    tf1 = 2/(1/tprecision + 1/trecall)

    # Write results
    outpath = os.path.join(test_directory, 'Analysis')
    out_file = os.path.join(outpath, 'threshold_metrics.txt')
    output = ['With Training Thresholds\n'
              'Number of true positives:\t{:d}\n'.format(tpos_all),
              'Number of false positives:\t{:d}\n'.format(fpos.sum()),
              'Number of false negatives:\t{:d}\n'.format(fneg.sum()),
              'Precision:                \t{:0.3f}\n'.format(precision),
              'Recall:                   \t{:0.3f}\n'.format(recall),
              'F1/Dice:                 \t{:0.3f}\n'.format(f1),
              '\n' + '-'*50 + '\n\n',
              'With Test Thresholds\n'
              'Number of true positives:\t{:d}\n'.format(ttpos_all),
              'Number of false positives:\t{:d}\n'.format(tfpos.sum()),
              'Number of false negatives:\t{:d}\n'.format(tfneg.sum()),
              'Precision:                \t{:0.3f}\n'.format(tprecision),
              'Recall:                   \t{:0.3f}\n'.format(trecall),
              'F1/Dice:                 \t{:0.3f}\n'.format(tf1),
              ]
    with open(out_file, 'w') as f:
        f.writelines(output)


if __name__ == "__main__":
    test_directory = '/media/matt/Seagate Expansion Drive/CT Data/LungTumors/Kfold_210315/test'
    train_thresh_file = '/home/matt/Documents/Projects/vnet_lung_tumors/data/kfold/train/Analysis/data_train/ideal_thresholds.txt'
    test_fname = 'data_5.json'
    compute_total_metrics_at_threshold(test_directory, test_fname, train_thresh_file)

    test_directory = '/media/matt/Seagate Expansion Drive/CT Data/LungTumors/Kfold_210315/test_xfer'
    train_thresh_file = '/home/matt/Documents/Projects/vnet_lung_tumors/data/kfold/train_xfer/Analysis/data_train/ideal_thresholds.txt'
    test_fname = 'data_5.json'
    compute_total_metrics_at_threshold(test_directory, test_fname, train_thresh_file)

    test_directory = '/media/matt/Seagate Expansion Drive/CT Data/LungTumors/Kfold_210315/test_sim_real'
    test_fname = 'data_5.json'
    train_thresh_file = '/home/matt/Documents/Projects/vnet_lung_tumors/data/kfold/train_for_sim/Analysis/data_train/ideal_thresholds.txt'
    calcuate_features(test_directory, test_fname)
    compute_total_metrics_at_threshold(test_directory, test_fname, train_thresh_file)
    detection_file = '/home/matt/Documents/Projects/vnet_lung_tumors/data/sim_only_test/Analysis/data.json'
    tumor_size_detection(detection_file, train_thresh_file)

    test_directory = '/media/matt/Seagate Expansion Drive/CT Data/LungTumors/Kfold_210315/sim_only_test'
    test_fname = 'data_5.json'
    train_thresh_file = '/home/matt/Documents/Projects/vnet_lung_tumors/data/kfold/train_for_sim/Analysis/data_train/ideal_thresholds.txt'
    calcuate_features(test_directory, test_fname)
    compute_total_metrics_at_threshold(test_directory, test_fname, train_thresh_file)


    train_thresh_file = '/home/matt/Documents/Projects/vnet_lung_tumors/data/kfold/train_xfer/Analysis/data_train/ideal_thresholds.txt'
    test_fname = 'data_4.json'
    calcuate_features(base_path, fname)
    compute_total_metrics_at_threshold(test_directory, test_fname, train_thresh_file)


    # Compute detection by size
    base_path = '/media/matt/Seagate Expansion Drive/CT Data/LungTumors/Kfold_210315/test_xfer'
    fname = 'kfold_sweep_detection.json'
    calcuate_features(base_path, fname)
    fname = 'data_5.json'
    k_fold_analysis(base_path, feature_name=fname)

    detection_file = '/media/matt/Seagate Expansion Drive/CT Data/LungTumors/Kfold_210315/test_xfer/Analysis/kfold_sweep_detection.json'
    train_thresh_file = '/home/matt/Documents/Projects/vnet_lung_tumors/data/kfold/train_xfer/Analysis/data_train/ideal_thresholds.txt'
    tumor_size_detection(detection_file, train_thresh_file)

    base_path = '/media/matt/Seagate Expansion Drive/CT Data/LungTumors/Kfold_210315/test'
    fname = 'kfold_sweep_detection.json'
    calcuate_features(base_path, fname)
    fname = 'data_5.json'
    k_fold_analysis(base_path, feature_name=fname)

    detection_file = '/media/matt/Seagate Expansion Drive/CT Data/LungTumors/Kfold_210315/test/Analysis/kfold_sweep_detection.json'
    train_thresh_file = '/home/matt/Documents/Projects/vnet_lung_tumors/data/kfold/train/Analysis/data_train/ideal_thresholds.txt'
    tumor_size_detection(detection_file, train_thresh_file)

