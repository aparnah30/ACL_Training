# OpenCV


## Projects Overview

### 1. **Oil Spill Segmentation using OpenCV**

This project focuses on detecting oil spills in satellite images(SAR) by identifying dark patches using simple image processing techniques with OpenCV.

- **Key Features**:
  - **Image Preprocessing**: Convert images to grayscale and resize them to a consistent size.
  - **Dark Patch Detection**: Apply a threshold to identify dark patches in the image, which represent oil spills.
  - **Visualization**: Display the original image, detected dark patches, and the corresponding mask side by side.

- **Resources Used**:
  - OpenCV
  - Pillow (PIL)
  - NumPy
  - Matplotlib

---

### 2. **VGG Architecture for Image Classification (PyTorch)**

This project implements a VGG-like architecture for image classification using PyTorch. The model is designed to take input images, apply convolutional layers, and output a class prediction.

- **Key Features**:
  - **Custom VGG Architecture**: Implements VGG16-like architecture with convolutional layers, max pooling, and fully connected layers.
  - **Forward Pass**: A standard forward pass that reshapes the output and applies fully connected layers to classify images.
  - **Model Validation**: The model is validated with a sample input tensor to ensure the network works as expected.

- **Resources Used**:
  - PyTorch
  - Torchvision

---

### 3. **Fine-Tuning a T5 Model for Medical Summarization**

This project fine-tunes a pre-trained T5 model from HuggingFace's `transformers` library to generate summaries for medical data. The T5 model is specifically fine-tuned on a medical dataset for summarization tasks.

- **Key Features**:
  - **Pre-trained T5 Model**: Leverages HuggingFace's `transformers` library to fine-tune the T5 model on a custom medical dataset.
  - **Summarization**: Input medical texts are summarized, allowing healthcare professionals to quickly grasp key information.
  - **HuggingFace Integration**: Easily integrates with the HuggingFace `transformers` library for fine-tuning and model management.

- **Resources Used**:
  - HuggingFace `transformers`
  - PyTorch

---

## Installation

To run this project, clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/your-username/repository-name.git
cd repository-name
pip install -r requirements.txt
