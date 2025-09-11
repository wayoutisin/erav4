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

## Training Results

```text
Model Parameter Breakdown:
Layer: conv1                | Parameters: 40
Layer: conv2                | Parameters: 296
Layer: conv3                | Parameters: 1168
Layer: conv4                | Parameters: 4640
Layer: fc1                  | Parameters: 10260
Layer: fc2                  | Parameters: 210

Total Trainable Parameters: 16614
Epoch 1, Batch 1, Records 0, Loss: 2.3051419258117676
Epoch 1, Batch 101, Records 10000, Loss: 0.14449019729811802
Epoch 1, Batch 201, Records 20000, Loss: 0.15814234057863553
Epoch 1, Batch 301, Records 30000, Loss: 0.10688325017690658
Epoch 1, Batch 401, Records 40000, Loss: 0.0815152737525357
Epoch 1, Batch 501, Records 50000, Loss: 0.09172087532394017
Accuracy on test data: 97.39%
```

The model achieves an accuracy of 97.39% on the test dataset after just 1 epoch of training. This is accomplished with a small model, consisting of approximately 16K trainable parameters.

### Test Evaluation Code

The `evaluate_accuracy` function, located in `train.py`, is responsible for calculating the model's accuracy on the test dataset. This function sets the model to evaluation mode (`model.eval()`), disables gradient calculation (`torch.no_grad()`) to save memory and computations, and then iterates through the test data to compare the model's predictions with the actual labels. The total number of correct predictions is then used to compute the overall accuracy.

```python
import torch

def evaluate_accuracy(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total
```

## Observations

1.  **Cross-Entropy and Softmax**: The provided code applied a softmax activation before computing the cross-entropy loss. However, PyTorch's `torch.nn.functional.cross_entropy` (and `nn.CrossEntropyLoss`) is designed to work with raw unnormalized logits. As such I had changed the output to only send logits.

2.  **Linear Layer Input Size**: The provided code specified an incorrect input size for the first fully connected layer (`fc1`). After the convolutional and pooling layers, the flattened feature map size was 512, not 320 as it was originally defined.

## Insights

To gain a deeper understanding of the dataset and how the model processes images, refer to the `mnist.ipynb` notebook. It contains several visualizations that illustrate:

1.  **Sample Images**: Random samples from the MNIST dataset are displayed, showing the handwritten digits and their corresponding labels. This helps in visualizing the raw input data the model is trained on.

2.  **Visualize Transformations**: This section demonstrates the output of intermediate layers of the `MNISTModel`. You can observe how an input image is transformed as it passes through the convolutional and max-pooling layers, revealing the feature maps extracted at each stage. This provides valuable insights into how the CNN learns to recognize patterns and features important for digit classification.

It is highly recommended to open `mnist.ipynb` to view these charts and outputs directly, as they offer an interactive and detailed perspective on the data and model's internal workings.
