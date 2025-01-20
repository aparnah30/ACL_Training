# Multi-Project Python Code

This repository contains three Python projects focused on different computer vision tasks: detecting dark patches in images for oil spill segmentation, person detection with YOLO, and pose estimation with MediaPipe. Each project demonstrates the application of various libraries in the field of computer vision.

## Projects Overview

### 1. Oil Spill Segementation in SAR images

This project focuses on detecting dark patches in images. It loads a grayscale image, resizes it, and applies a threshold to detect areas of low intensity (dark patches). The project visualizes the results, displaying the original image, the image with detected dark patches, and the mask used for detection.

#### Key Features:
- Loads and preprocesses images.
- Detects dark patches using a pixel intensity threshold.
- Displays results using Matplotlib.

#### Libraries Used:
- `torch`
- `torchvision`
- `PIL` (Python Imaging Library)
- `numpy`
- `matplotlib`

---

### 2. Person Detection with YOLO

This project leverages the YOLO (You Only Look Once) object detection model to identify and count people in real-time using a webcam feed. It utilizes the pre-trained YOLOv8 model and draws bounding boxes around detected people. The system continuously counts and displays the number of people detected in the webcam frame.

#### Key Features:
- Real-time object detection using YOLOv8.
- Counts and displays the number of people detected in the webcam feed.
- Displays confidence and class names for detected objects.

#### Libraries Used:
- `ultralytics` (for YOLO model)
- `opencv-python`
- `math`

#### Setup Instructions:
1. Install the required dependencies:
   ```bash
   pip install opencv-python ultralytics
