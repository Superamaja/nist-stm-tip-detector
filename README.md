# STM Tip Detector

A machine learning solution to detect if a STM tip is sharp or dull given a STM image. It uses a convolutional neural network to make predictions on the sharpness of the tip through the extraction of dangling bond features from the STM image.

## Table of Contents

-   [Getting Started](#getting-started)
    -   [Installation](#installation)
    -   [Usage](#usage)
-   [Program Overview](#program-overview)
-   [Configurations](#configurations)
-   [Additional Information](#additional-information)

## Getting Started

### Installation

1. Clone the repository:

```bash
git clone https://github.com/Superamaja/stm-tip-detector
```

2. Install the required packages:

```bash
pip install -r requirements.txt
```

### Usage

Run any python file in the root directory to start a program:

```bash
python <program>.py
```

### API Usage

The programs that can be run are listed below in the [Program Overview](#program-overview) section.

For sole usage of the API, copy the following files to your project:

-   [detector_functions/](detector_functions/)
-   [api.py](api.py)
-   [config.json](config.json)
-   [model.h5](model.h5) - **NOT PROVIDED HERE**

Then start the API:

```bash
python api.py
```

## Program Overview

**api.py** - The API that allows for external interfaces to interact with the software to make predictions. Real examples for the JSON input and outputs are in the [/api_examples](api_examples) folder. The full list of parameters can be found at the top of [api.py](api.py). The API serves to `localhost:5050` by default.

The following flags can be used to run the API:

-   `--host <host>` - The host to run the API on. Default is `localhost`.
-   `--port <port>` - The port to run the API on. Default is `5050`.

**detector.py** - An interface for the software to make predictions on STM images. Allows for efficient scanning of STM images to test models, along with useful debugging windows. Saves the configurations of the STM images (contrast, rotation, etc) to speed up the process of testing models. Exports a `results.csv` file that contains information about the output.

The following flags can be used to run the detector:

-   `-sd` - Activates the scan debug mode, which will display the scanned STM image along with other useful windows and information.
-   `-rd` - Activates the ROI debug mode, which will display each extracted ROI from the STM image. Useful for visually debugging the ROI that gets passed to the model.

**processor.py** - A program designed to process STM images and extract the dangling bond features from them to ensure consistency in the data along with matching the detector's extraction process for predictions.

The following flags can be used to run the processor:

-   `-d` - Activates the debug mode, which will display each training image along with the extraction of each ROI. Useful for visually debugging the ROI that becomes the new training data.

-   `-s <start_index>` - Start index for which image to start processing. Useful for debugging at a certain point in the dataset.

-   `-i <image_directory>` - The image directory to process. Overrides the default image directory in a variable at the top of the file.

**trainer.py** - A program designed to train a machine learning model on the extracted features from the STM images. Comes with detailed logging for efficient model fine-tuning.

## Configurations

**config.json** - A configuration file that contains all the hyperparameters for the model. Used for global variables that need to be consistent across all components of the software. Also includes variables to speed up the process of creating and testing models.

-   `ROI_NM_SIZE` - The size of the ROI in nanometers. Used to extract the ROI for processing training data and detecting sharpness on STM images.
-   `SQUARE_PIXEL_SIZE` - The pixel size of the extracted ROI. Used to resize the ROI to a consistent size for the model.
-   `DETECTOR_CROSS_SIZE` - The size of the cross that is used to "cross scan" a detected ROI. Essentially creates vertical and horizontal shifts to a detected ROI and uses the max prediction as the final prediction.
-   `DETECTOR_SCAN_DEBUG` - A variable to activate the scan debug mode in the detector. See the detector.py `-sd` section for more information.
-   `DETECTOR_ROI_DEBUG` - A variable to activate the ROI debug mode in the detector. See the detector.py `-rd` section for more information.
-   `SCANS` - A list of STM images to scan for predictions. Gets looped through in the detector to make predictions on each image.

**configs/scan_configs.json** - An automatically generated configuration file that contains the contrast, rotation, and other configurations for each STM image. Used for [detector.py](detector.py) to speed up the process of testing models.

## Additional Information

This project was developed by [Connor Lin](https://github.com/Superamaja/) during his 2024 [NIST](https://www.nist.gov/) internship under the mentorship of [Dr. Jonathan Wyrick](https://github.com/juanquij0te).

It was developed to serve as the foundation for future computer vision involved STM automation projects at NIST.

The research poster for this project can be found [here].
