# MNIST Handwritten Digit Recognition

## Overview
This project implements a Convolutional Neural Network (CNN) for handwritten digit recognition using the MNIST dataset. The implementation is done in PyTorch and demonstrated through a Jupyter notebook, `mnist.ipynb`. The notebook covers data loading, model definition, training, and visualization of the model's intermediate layers.

## Files
- `dataset.py`: Defines the `MNISTDataset` custom PyTorch Dataset for loading and transforming MNIST images.
- `model.py`: Defines the `MNISTModel` CNN architecture.
- `train.py`: Contains the training and evaluation logic (this code is integrated into `mnist.ipynb`).
- `mnist.ipynb`: A Jupyter notebook that consolidates the code from `dataset.py`, `model.py`, and `train.py`, with added markdown explanations and visualizations.
- `README.md`: This file.

## Model Architecture
The `MNISTModel` is a simple yet effective Convolutional Neural Network for image classification. It consists of the following layers:

1.  **Convolutional Layer 1 (`conv1`)**: `nn.Conv2d(1, 4, kernel_size=3)`
    -   Input: 1-channel grayscale image (28x28)
    -   Output: 4 feature maps (26x26) after ReLU activation.

2.  **Convolutional Layer 2 (`conv2`)**: `nn.Conv2d(4, 8, kernel_size=3)`
    -   Input: 4 feature maps (26x26)
    -   Output: 8 feature maps (24x24) before max-pooling.

3.  **Max Pooling Layer**: `nn.MaxPool2d(kernel_size=2)`
    -   Reduces spatial dimensions (e.g., 24x24 to 12x12), preserving important features.

4.  **Convolutional Layer 3 (`conv3`)**: `nn.Conv2d(8, 16, kernel_size=3)`
    -   Input: 8 feature maps (12x12)
    -   Output: 16 feature maps (10x10) after ReLU activation.

5.  **Convolutional Layer 4 (`conv4`)**: `nn.Conv2d(16, 32, kernel_size=3)`
    -   Input: 16 feature maps (10x10)
    -   Output: 32 feature maps (8x8) after ReLU activation.

6.  **Max Pooling Layer**: `nn.MaxPool2d(kernel_size=2)`
    -   Reduces spatial dimensions (e.g., 8x8 to 4x4).

7.  **Flatten Layer**: The output from the last max-pooling layer (4x4x32) is flattened into a 1D vector (512 features).

8.  **Fully Connected Layer 1 (`fc1`)**: `nn.Linear(512, 20)`
    -   Input: 512 flattened features.
    -   Output: 20 features after ReLU activation.

9.  **Fully Connected Layer 2 (`fc2`)**: `nn.Linear(20, 10)`
    -   Input: 20 features.
    -   Output: 10 features, corresponding to the 10 MNIST digit classes (0-9).


## How to Run
1.  **Ensure Dependencies**: Make sure you have PyTorch, `torch`, `torchvision`, `datasets`, and optionally `matplotlib` installed (if running the notebook). You can install them using pip:
    ```bash
    pip install torch torchvision datasets matplotlib
    ```
2.  **Open Terminal**: Navigate to the `wk4-mnist` directory.
    ```zsh
    uv run train.py
    ```