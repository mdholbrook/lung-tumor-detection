import os
from glob2 import glob
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def load_filenames(base_path):
    # Get filenames - Tumor
    im_files = os.path.join(base_path, '**/label.nii')
    im_files = glob(im_files)


    return im_files


def measure_connected(im):

    # Compute measured components
    con_im = sitk.ConnectedComponent(im)

    # Compute region stats
    reg_stats = sitk.LabelShapeStatisticsImageFilter()
    reg_stats.Execute(con_im)

    # Get statistics
    voxels = []
    for label in range(1, reg_stats.GetNumberOfLabels()+1):
        voxels.append(reg_stats.GetNumberOfPixels(label))

    return voxels


def compute_histogram_of_volumes():
    base_path = '/media/matt/Seagate Expansion Drive/CT Data/LungTumors/Kfold_210315/test_xfer'
    sub_paths = glob(os.path.join(base_path, '??/'))
    num_voxels = []
    n = 0
    for path in sub_paths:
        print(f'Working on {path}')
        ims = load_filenames(path)

        for im_name in ims:
            # Load image
            im = sitk.ReadImage(im_name)

            # Compute regions
            nvoxels = measure_connected(im)
            num_voxels += nvoxels

    # Compute histogram
    fig = plt.figure(1)
    plt.hist(num_voxels, bins=20, range=[0, 10000])
    plt.xlabel('Label Volume [voxels]')
    plt.show()


def original_data():
    base_path = '/home/matt/Documents/Projects/vnet_lung_tumors/data/Real_xfer_210305'
    ims = sorted(glob(os.path.join(base_path, '**/label.nii*')))
    num_voxels = []

    for im_name in ims:
        # Load image
        im = sitk.ReadImage(im_name)

        # Compute regions
        nvoxels = measure_connected(im)
        print(f'{im_name}: {len(nvoxels)}')
        num_voxels += nvoxels



