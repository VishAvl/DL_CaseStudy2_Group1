# VGG19 Face Expression Classification

This repository contains the implementation of a face expression classification model using the VGG19 architecture. The model is trained to classify images into six categories: neutral, sad, happy, angry, surprise, and ahegao.

## Table of Contents

- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Evaluation](#evaluation)
- [Saving the Model](#saving-the-model)
- [Results](#results)
- [License](#license)

## Dataset

The dataset used for training the model consists of images labeled with six different face expressions. The dataset is provided in a CSV file with two columns:
- `path`: Path to the image file
- `label`: The label for the image (neutral, sad, happy, angry, surprise, ahegao)

## Model Architecture

The model is based on the VGG19 architecture, pretrained on the ImageNet dataset. The base model is used as a feature extractor, and additional layers are added on top for classification.

The model architecture:
- Base Model: VGG19 (pretrained, with top layers removed)
- Global Average Pooling Layer
- Dense Layer (1024 units, ReLU activation)
- Batch Normalization
- Dropout (0.5)
- Dense Layer (512 units, ReLU activation)
- Batch Normalization
- Dropout (0.5)
- Output Layer (6 units, Softmax activation)

## Installation

To install the required packages, run:

```bash
pip install tensorflow pandas scikit-learn matplotlib seaborn
```

## Usage

1. Clone the repository:

```bash
git clone https://github.com/your-username/face-expression-classification.git
cd face-expression-classification
```

2. Place your dataset CSV file in the root directory of the repository.

## Training

To train the model, run:

```bash
python model.py
```

This will load the dataset, preprocess the images, build and compile the model, and start the training process. The best model based on validation loss will be saved as `best_model.h5`.

## Evaluation

After training, the model is evaluated using confusion matrix and ROC-AUC curves. The evaluation metrics are plotted to help you analyze the model's performance.

## Saving the Model

The entire model, including the architecture, weights, and optimizer state, is saved as `model_with_weights.h5` after training. This allows you to reload and use the model later without needing to recompile it.

## Results

- The model is evaluated on validation data, and the confusion matrix and ROC-AUC curves are plotted.
- The training time and accuracy are also printed.

## License

This project is licensed under the MIT License.

---

Feel free to customize the `README.md` file as per your requirements.
