# transfer_learning_fashion

# Fashion-MNIST Image Classification Using Transfer Learning (VGG16, PyTorch)

## Project Overview

This project implements an image classification pipeline for the Fashion-MNIST dataset using **transfer learning** with a pretrained **VGG16** model in PyTorch. The goal is to leverage visual features learned from a large dataset (ImageNet) to improve classification accuracy on a smaller fashion image dataset, reducing training time and data requirements.

## Dataset

- **Fashion-MNIST** consists of 28x28 grayscale images of 10 clothing categories (e.g., T-shirts, trousers, sneakers).
- Data is provided in CSV format, each row containing a label and 784 pixel values flattened from 28x28 images.

## Methodology

1. **Data Loading and Visualization:**
   - Data is loaded using pandas.
   - Images are reshaped from flat vectors and visualized for exploratory analysis.

2. **Preprocessing:**
   - Grayscale images are converted to 3-channel RGB.
   - Images are resized to 224x224 pixels to match VGG16 input requirements.
   - Normalization is applied using ImageNet statistics.

3. **Custom PyTorch Dataset and DataLoader:**
   - Efficient batching and transformation occur via custom Dataset classes.

4. **Model Setup:**
   - VGG16 pretrained on ImageNet is loaded.
   - Feature extraction layers are frozen to retain learned weights.
   - The classifier head is replaced with fully connected layers adapted for 10 classes, including dropout for regularization.

5. **Training:**
   - Only the classifier layers are trained using the Adam optimizer and cross-entropy loss.
   - Training is performed on GPU if available for acceleration.
   - Training loss is tracked across epochs.

6. **Evaluation:**
   - Model accuracy is evaluated on both training and test sets.
   - Training loss curve visualization is included for insight into training progression.

## Highlights

- Demonstrates effective **transfer learning**, adapting a complex model to a domain-specific task.
- Utilizes **GPU acceleration** to optimize training performance.
- Includes an end-to-end pipeline from data preprocessing to model evaluation.
- Provides a modular, well-documented, and reproducible approach suitable for portfolios.

## Potential Extensions

- Implement baseline models (e.g., simple CNN from scratch) to benchmark performance.
- Explore hyperparameter tuning and data augmentation.
- Add detailed metrics such as confusion matrices and classification reports.
