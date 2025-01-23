# Pneumonia Detection Using VGG16

This project implements a deep learning model for detecting pneumonia in chest X-ray images using the VGG16 architecture. The model is trained to classify X-ray images into two classes: **Pneumonia** and **Normal**. This is a binary classification task where we use the popular VGG16 model architecture as a backbone.

## Key Features

- **VGG16 Architecture**: VGG16 model implemented in PyTorch with convolutional layers, batch normalization, ReLU activation, and fully connected layers.
- **Transfer Learning**: Uses pre-trained weights and fine-tunes the model for pneumonia detection.
- **Data Augmentation**: X-ray images are resized, normalized, and converted to tensor format for model compatibility.
- **Optimized Training**: The model is trained using the Adam optimizer and CrossEntropyLoss for classification.
- **Evaluation**: Model performance is evaluated using classification metrics like accuracy, precision, recall, and confusion matrix.

## Requirements

- Python 3
- PyTorch
- torchvision
- scikit-learn
- numpy
- matplotlib (optional for visualization)

## Installation

Clone the repository:

```bash
git clone https://github.com/aparnah30/ACL_Training.git
cd deepLearning
