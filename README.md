# Brain Tumor Segmentation using U-Net

## Overview
This project implements a 2D U-Net for Brain Tumor Segmentation based on the BraTS dataset. It processes 4 MRI modalities (T1, T1c, T2, FLAIR) to perform segmentation tasks, in this case we are only targeting the whole tumor (WT). The model has been adapted for 2D inputs instead of 3D volumes, focusing on slicing the images along the axial plane.

## Conda Environment Setup
The environment is managed using `conda`. To set up the environment, follow these steps:

1. Create the environment using the `AEP.yml` file:
   ```bash
   conda env create -f AEP.yml

## Dataset
The project expects the dataset to be organized in the BraTS format. Each subject's data should be stored in a directory under the main dataset folder, containing the following modalities:
- **T1-weighted (T1)**
- **Gadolinium-enhanced T1-weighted (T1c)**
- **T2-weighted (T2)**
- **Fluid Attenuated Inversion Recovery (FLAIR)**
- **Segmentation mask (seg)**

## Code Structure

### Data Preprocessing
- **Loading the dataset:** The MRI modalities and segmentation masks are loaded using NiBabel. Data from a single patient is used for demonstration purposes.
- **Cropping:** The input images and masks are cropped to focus on the tumor area, eliminating most background noise.
- **Resizing:** Volumes are resized to 128x128x128.
- **Slicing:** The 3D volumes are split into 2D axial slices, where each slice becomes an independent training sample.

### Model Architecture
The model is based on the U-Net architecture, which consists of:
- **Encoder:** The encoder uses convolutional layers followed by max-pooling to reduce the spatial dimensions and capture feature representations.
- **Decoder:** The decoder uses upsampling and skip connections to restore the spatial dimensions, aiming to produce a segmentation mask that matches the original input size.
- **Dilation layers:** Additional dilated convolution layers are used in the bottleneck part of the U-Net, following the implementation in the BraTS 2020 paper.

### Loss Function
The model uses a Dice Loss function, which is specifically designed for segmentation tasks and works well with imbalanced datasets. It is implemented to measure the overlap between predicted and ground truth segmentations.

### Knowledge Distillation
A teacher election mechanism is implemented to perform Knowledge Distillation, where the best modality (among T1, T1c, T2, and FLAIR) is chosen based on the performance of the network on validation data.

### Training
The training loop involves:
- **Shuffling the dataset** and splitting it into training, validation, and test sets.
- **Optimization:** The Adam optimizer is used, along with a learning rate scheduler that reduces the learning rate after a certain number of steps.
- **Training epochs:** The model trains for a specified number of epochs (1000 by default), where the Dice Loss is monitored and printed during training.

## Usage

1. **Prepare the dataset:**
   Place the dataset in a directory structure where each subject folder contains the following files: T1, T1c, T2, FLAIR, and segmentation masks.

2. **Run the script:**
   Execute the script by running:
   ```bash
   jupyter notebook

3.Open the Notebook: Once Jupyter launches in your default browser, navigate to the directory where your script is located, and open the desired .ipynb notebook.
   
