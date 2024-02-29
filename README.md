
# MIDiffusion
PyTorch implementation of Mutual Information Guided Diffusion for Zero-shot Cross-modality Medical Image Translation

`MIDiffusion` is a framework designed for zero-shot learning cross modality medical image translation based on diffusion models.

Zihao Wang, Yingyu Yang, Yuzhou Chen, Tingting Yuan, Maxime Sermesant, Hervé Delingette, Ona Wu

| Harvard | Temple | Univ. Göttingen | Inria |

## Overview
Utilizing tools for multi-modal data management offers advantages and challenges. In medical imaging, tools exist for segmenting images in modalities like 3D T1-weighted MRI, but are limited for others like Proton Density-weighted MRI. Using T1w segmentation tools on PDw images could be more efficient than seeking PDw-specific tools. A solution is zero-shot cross-modality image translation, using a diffusion model based on statistical homogeneity to bridge modalities. This approach, illustrated in our model (Locale-based Mutual Information), enables unsupervised zero-shot translation for effective cross-modality segmentation.

Schematic diagram shows the LMI-guided diffusion for zero-shot cross-modal segmentation. The blue and orange contours are source and target distributions. The blue dot in the orange contour represents the target datapoint of the source datapoint (orange dot in the blue contour) in the source distribution. LMIDiffusion uses explicit statistical features (LMI) to navigate the next step (yellow dot), providing continuous guidance (yellow dot) from start to finish. In the end, the translated image can be segmented using arbitrary segmentation methods that were trained only on the target modality.

![Caption.](./others/condition_ablation.png)

## Project Structure

- `configs/`: Configuration files for training and evaluation.
- `datasets/`: Place your dataset files here or scripts to download datasets.
- `functions/`: Core functionalities including model components and utilities.
- `models/`: Definitions of diffusion models and architectures.
- `runners/`: Scripts for different stages of the experiment lifecycle.
- `utils/`: Utility functions and helpers.


## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/midiffusion-main.git
cd midiffusion-main
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Training

To start a training session, you can use the provided shell scripts:

```bash
./train_commands.sh
```

The command line for training with `main.py` script in `train_commands.sh` includes several options that allow you to customize the training process. Here's a detailed explanation of each argument used in the command:

```bash
python main.py --use_unet --config ./configs/CuRIOUS_FLAIR_T1.yml --gpu_id 0 --seed 1234 --comment "" --verbose info --image_folder images --exp CuRIOUST1 --doc doc --train_path_a /CuRIOUS/imagesTr_slices/train/T1 --train_path_b /CuRIOUS/imagesTr_slices/train/T1
```


- `--use_unet`: This flag indicates that the U-Net architecture should be used for the model. U-Net is particularly popular for medical image segmentation tasks due to its effectiveness in handling such data.
- `--config ./configs/CuRIOUS_FLAIR_T1.yml`: Specifies the path to the configuration file that contains detailed settings for the training process. This file is essential for setting up the experiment parameters.
- `--gpu_id 0`: Assigns the ID of the GPU to be used for training. This is important for systems with multiple GPUs.
- `--seed 1234`: Sets the random seed to `1234` to ensure reproducibility of the results. This is crucial for scientific experiments where reproducibility is key.
- `--comment ""`: Allows for an optional comment string to be associated with the training run. In this case, it's left empty.
- `--verbose info`: Sets the verbosity level of the output to `info`. This controls how much information is printed out during the training process. Other levels include `debug`, `warning`, and `critical`.
- `--image_folder images`: Designates the folder where output images (e.g., training progress snapshots) will be saved.
- `--exp CuRIOUST1`: Specifies the name of the experiment. This can be useful for organizing and identifying different training runs.
- `--doc doc`: Provides a string for documentation purposes, which will be used as the name of the log folder. This helps in keeping track of different experiments and their outcomes.
- `--train_path_a /CuRIOUS/imagesTr_slices/train/T1`: Specifies the path to the training set for modality A. In this context, it points to the location of T1-weighted MRI slices used for training.
- `--train_path_b /CuRIOUS/imagesTr_slices/train/T1`: Specifies the path to the training set for modality B. Even though it points to the same T1-weighted MRI slices as modality A in this example, typically, this could be used to specify a different modality or data set for comparative training or multi-modal training scenarios.


For training with specific settings, modify the `train_commands.sh` script or run `main.py` with custom arguments.

### Sampling
#### Command Line Usage for run_sampling
To generate samples from a trained model:

```bash
./run_sampling.sh
```

The following command initiates a sampling process using the `Sampling.py` script in `run_sampling.sh`, demonstrating its usage with a comprehensive set of arguments for a specific task:

```bash
python Sampling.py --use_unet --config ./configs/IXI.yml --seed 1234 --comment "" --exp CuRIOUST1_IXIPD/ --train_path_a /CuRIOUS/imagesTr_slices/test/T1 --train_path_b /CuRIOUS/imagesTr_slices/test/T2 --verbose info --image_folder images_level500 --doc doc --sample --use_pretrained --fid --sample_step 3 --t 1000
```

### Argument Breakdown

- `--use_unet`: Utilizes a U-Net architecture for the model. U-Net is widely used for image segmentation tasks.
- `--config ./configs/IXI.yml`: Path to the configuration file containing experiment parameters.
- `--seed 1234`: Sets the random seed to `1234` for reproducible results.
- `--comment " "`: Allows for an optional comment about the experiment. In this case, it is left empty.
- `--exp CuRIOUST1_IXIPD/`: Designates the directory for saving experiment-related data.
- `--train_path_a /CuRIOUS/imagesTr_slices/test/T1`: Specifies the path to the training set for modality A.
- `--train_path_b /CuRIOUS/imagesTr_slices/test/T2`: Specifies the path to the training set for modality B.
- `--verbose info`: Sets the verbosity level to `info` for logging information.
- `--image_folder images_level500`: The directory where sampled images will be stored.
- `--doc doc`: Names the log folder for documentation purposes, aiding in distinguishing between experiment runs.
- `--sample`: Activates the model's sampling mode to generate samples.
- `--use_pretrained`: Indicates that sampling will utilize a pretrained model.
- `--fid`: Triggers the calculation of the Frechet Inception Distance (FID) for evaluating the quality of the generated images.
- `--sample_step 3`: Defines the total number of steps in the sampling process.
- `--t 1000`: Specifies the noise scale used in sampling, affecting the variance of the generated samples.

## References

This implementation is based on / inspired by:

https://github.com/ermongroup/ddim

https://github.com/ermongroup/SDEdit

https://github.com/ermongroup/ncsnv2

https://github.com/pesser/pytorch_diffusion

