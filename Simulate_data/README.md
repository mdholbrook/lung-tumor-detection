# Synthetic dataset generation

To generate realistic lung tumors, scans of healthy mice and segmented lung tumors are randomly combined to create a large number of unique sets. The tumors are warped and retextured to boost the variety of tumors seen during network training.

## Explanation of code

This code was written in MATLAB, and takes advantage of the GPU computing toolkit which has been developed at the Quantitative Image and Analysis Lab. To run this code as is one would need to request access to this toolkit.

### Lung mask generation

The first processing step for both sets both with and without lung tumors is to create a lung mask. This is performed by the `make_lung_mask.m` function. This function thresholds images and processes the resulting low-density regions to get an accurate lung segmenation. An exmple of how this script is called is found in `Create_lung_masks.m`.

### Creating simulated tumor sets

The generation of synthetic sets requires both the starting image and a bank of segmented lung tumors. These tumors are modified and placed in the lungs at random spots, with increased probability at the lung boundaries as is seen in real mice. This process is performed with the `Gen_lung_tumor_sims.m` script. Many of the functions found in this folder are called by this script.

The resulting data is sorted and prepared for network processing by `Sort_training_data.m`.
