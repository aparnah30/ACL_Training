import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def load_image(image_path):
    img = Image.open(image_path).convert('L')  
    return img

def resize_image(img, size=(256, 256)):
    resize_transform = transforms.Resize(size)
    img_resized = resize_transform(img)
    return img_resized

def detect_dark_patches(img, threshold=60):
    img_array = np.array(img)
    
    mask = img_array < threshold  
    
    img_detected = np.copy(img_array)
    img_detected[mask] = 0 
    
    return mask, img_detected

def plot_results(img, img_detected, mask):
    plt.figure(figsize=(10, 5))

    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    # Detected dark patches
    plt.subplot(1, 3, 2)
    plt.imshow(img_detected, cmap='gray')
    plt.title('Detected Dark Patches')
    plt.axis('off')

    # Mask visualization
    plt.subplot(1, 3, 3)
    plt.imshow(mask, cmap='gray')
    plt.title('Mask for Dark Patches')
    plt.axis('off')

    plt.show()

def preprocess_image(image_path):
    img = load_image(image_path)
    img_resized = resize_image(img)
    mask, img_detected = detect_dark_patches(img_resized)
    plot_results(img_resized, img_detected, mask)

    return img_resized, mask

image_path = 'data/Samples/1_200_0_img_dtsR0emPVPPNdxSS_SFr_cls_1.jpg'
img_resized, dark_patch_mask = preprocess_image(image_path)
