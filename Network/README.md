# Tensorflow implementation of V-Net

The network employed here is adapted from a pre-existing implementation of V-Net found here. This is a Tensorflow implementation of the [V-Net](https://arxiv.org/abs/1606.04797) architecture used for 3D medical imaging segmentation. This code adopts the tensorflow graph from [here](https://github.com/jackyko1991/vnet-tensorflow). The repository covers k-fold training, evaluation and prediction modules for the (multimodal) 3D medical image segmentation in multiple classes.

## Notes on training and processing

This repository is dependent on Tensorflow 1.14. The easiest way to run this  on most computers requires a Docker container. After installing docker and the NVidia container for running CUDA, an image can be set up for training and evaluation.

### Setting up a Docker image

The setup parameters are found in `dockerfile` which downloads the correct container and installs the prerequisites. This file can be run from the command line using `docker build -t lungdl .` or by modifying the bash file `docker_run.sh` by commenting out the "Build" section.

### Running the network

To run a specific training or evaluation copy and modify one of the `.json` parameter files. To train, you will need to modify the location of the training data and the path which saves the model and training metrics. For evaluation, the path to the model and to the data must be modified. The output is saved in the same path as the data. Examples of how to run a training or evaluation are located in `docker_run.sh`.

To run batches, such as for k-folds, the files named `kfold*.py` set up the training data, create new configuration files for each fold, and run the networks in the docker container. These files are run from python without the use of docker. Multiple networks can be trained at the same time by adjusting the `num_gpus` parameter, though training more than 3 slows the process down due to disk I/O operations.

### Visual Representation of Network

Here is an example graph of network this code implements. Channel depth may change owning to change in modality number and class number.
![VNetDiagram](VNetDiagram.png)

### Features

- 2D and 3D data processing ready
- Augmented patching technique, requires less image input for training
- Multichannel input and multiclass output
- Generic image reader with SimpleITK support (Currently only support .nii/.nii.gz format for convenience, easy to expand to DICOM, tiff and jpg format)
- Medical image pre-post processing with SimpleITK filters
- Easy network replacement structure
- Sørensen and Jaccard similarity measurement as golden standard in medical image segmentation benchmarking
- Utilizing medical image headers to retrive space and orientation info after passthrough the network

## Usage

### Required Libraries

Known good dependencies

- Python 3.7
- Tensorflow 1.14 or above
- SimpleITK

### Folder Hierarchy

All training, testing and evaluation data should put in `./data`

    .
    ├── ...
    ├── data                      # All data
    │   ├── testing               # Put all testing data here
    |   |   ├── case1            
    |   |   |   ├── img.nii.gz    # Image for testing
    |   |   |   └── label.nii.gz  # Corresponding label for testing
    |   |   ├── case2
    |   |   ├──...
    │   ├── training              # Put all training data here
    |   |   ├── case1             # foldername for the cases is arbitary
    |   |   |   ├── img.nii.gz    # Image for training
    |   |   |   └── label.nii.gz  # Corresponding label for training
    |   |   ├── case2
    |   |   ├──...
    │   └── evaluation            # Put all evaluation data here
    |   |   ├── case1             # foldername for the cases is arbitary
    |   |   |   └── img.nii.gz    # Image for evaluation
    |   |   ├── case2
    |   |   ├──...
    ├── tmp
    |   ├── cktp                  # Tensorflow checkpoints
    |   └── log                   # Tensorboard logging folder
    ├── ...

If you wish to use image and label with filename other than `img.nii.gz` and `label.nii.gz`, please change the following values in `config.json`

    "ImageFilenames": ["img.nii.gz"],
    "LabelFilename": "label.nii.gz"

The network will automatically select 2D/3D mode by the length of `PatchShape` in `config.json`

In segmentation tasks, image and label are always in pair, missing either one would terminate the training process.

The code has been tested with [LiTS dataset](http://academictorrents.com/details/27772adef6f563a1ecc0ae19a528b956e6c803ce)

### Training

You may run train.py with commandline arguments. To check usage, type ```python main.py -h``` in terminal to list all possible training parameters.

Available training parameters

      -h, --help            show this help message and exit
      -v, --verbose         Show verbose output
      -p [train evaluate], --phase [train evaluate]
                            Training phase (default= train)
      --config_json FILENAME
                            JSON file for model configuration
      --gpu GPU_IDs         Select GPU device(s) (default = 0)

The program will read the configuration from `config.json`. Modify the necessary hyperparameters to suit your dataset.

Note: You should always set label 0 as the first `SegmentationClasses` in `config.json`. Current model will only run properly with at least 2 classes.

The software will automatically determine run in 2D or 3D mode according to rank of `PatchShape` in `config.json`

#### Image batch preparation

Typically medical image is large in size when comparing with natural images (height x width x layers x modality), where number of layers could up to hundred or thousands of slices. Also medical images are not bounded to unsigned char pixel type but accepts short, double or even float pixel type. This will consume large amount of GPU memories, which is a great barrier limiting the application of neural network in medical field.

Here we introduce serveral data augmentation skills that allow users to normalize and resample medical images in 3D sense. In `train.py`, you can access to `trainTransforms`/`testTransforms`. For general purpose we combine the advantage of tensorflow dataset api and SimpleITK (SITK) image processing toolkit together. Following is the preprocessing pipeline in SITK side to facilitate image augmentation with limited available memories.

1. Image Normalization (fit to 0-255)
2. Isotropic Resampling (adjustable size, in mm)
3. Padding (allow input image batch smaller than network input size to be trained)
4. Random Crop (randomly select a zone in the 3D medical image in exact size as network input)
5. Gaussian Noise

The preprocessing pipeline can easily be adjusted with following example code in `train.py`:

    trainTransforms = [
                    NiftiDataset3D.Normalization(),
                    NiftiDataset3D.Resample(self.spacing[0],self.spacing[1],self.spacing[2]),
                    NiftiDataset3D.Padding((self.patch_shape[0], self.patch_shape[1], self.patch_shape[2])),
                    NiftiDataset3D.RandomCrop((self.patch_shape[0], self.patch_shape[1], self.patch_shape[2]),self.drop_ratio,self.min_pixel),
                    NiftiDataset3D.RandomNoise()
                    ]

For 2D image training mode, you need to provide transforms in 2D and 3D separately.

To write you own preprocessing pipeline, you need to modify the preprocessing classes in `NiftiDataset.py`

Additional preprocessing classes:

- StatisticalNormalization
- Reorient (take care on the direction when performing evaluation)
- Invert
- ConfidenceCrop (for very small volume like cerebral microbleeds, alternative of RandomCrop)
- Deformations:
  The deformations are following SITK deep learning data augmentation documentations, will be expand soon.
  Now contains:
  - BSplineDeformation

  **Hint: Directly apply deformation is slow. Instead you can first perform cropping with a larger than patch size region then with deformation, then crop to actual patch size. If you apply deformation to exact training size region, it will create black zone which may affect final training accuracy.**
  
  Example:

      NiftiDataset.ConfidenceCrop((self.patch_shape[0]*2, self.patch_shape[1]*2, self.patch_shape[2]*2),(0.0001,0.0001,0.0001)),
      NiftiDataset.BSplineDeformation(),
      NiftiDataset.ConfidenceCrop((self.patch_shape[0], self.patch_shape[1], self.patch_shape[2]),(0.5,0.5,0.25)),

#### Tensorboard

In training stage, result can be visualized via Tensorboard. Run the following command:

    tensorboard --logdir=./tmp/log

Once TensorBoard is running, navigate your web browser to ```localhost:6006``` to view the TensorBoard.

Note: ```localhost``` may need to change to localhost name by your own in newer version of Tensorboard.

### Evaluation

To evaluate image data, first place the data in folder ```./data/evaluate```. Each image data should be placed in separate folder as indicated in the folder hierarchy

There are several parameters you need to set in order manually

- `model_path`, the default path is at `./tmp/ckpt/checkpoint-<global_step>.meta`
- `checkpoint_dir`, the default path is at `./tmp/ckpt`
- `patch_size`, this value need to be same as the one used in training
- `patch_layer`, this value need to be same as the one used in training
- `stride_inplane`, this value should be <= `patch_size`
- `stride_layer`, this value should be <= `patch_layer`
- `batch_size`, currently only support single batch processing

Run `evaluate.py` after you have modified the corresponding variables. All data in `./data/evaluate` will be iterated. Segmented label is named as `label_vnet.nii.gz` in same folder of the respective `img.nii.gz`.

You may change output label name by changing the line `writer.SetFileName(os.path.join(FLAGS.data_dir,case,'label_vnet.nii.gz'))`

Note that you should keep preprocessing pipeline similar to the one in `train.py`, but without random cropping and noise.
