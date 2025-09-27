import torch # Import the PyTorch library for building neural networks
import torch.nn as nn # Import the neural network module from PyTorch

class MNISTModelFinal(nn.Module):
    """Your model with configurable dropout rates"""
    # This class defines a convolutional neural network (CNN) for MNIST digit classification.
    # It inherits from nn.Module, the base class for all neural network modules in PyTorch.
    def __init__(self):
        # Call the constructor of the parent class (nn.Module)
        super(MNISTModelFinal, self).__init__()

        # Define the convolutional layers using nn.Sequential
        # nn.Sequential is a container that holds a sequence of modules.
        # The input will be passed through all modules in the same order as they are added.
        self.conv1 = nn.Sequential(
            # First Convolutional Block
            nn.Conv2d(1, 12, kernel_size=3, bias=False),  # 2D Convolution: 1 input channel (grayscale), 12 output channels, 3x3 kernel
            nn.BatchNorm2d(12),  # Batch Normalization: normalizes outputs for stable training
            nn.ReLU(inplace=True),  # ReLU activation: introduces non-linearity

            # Second Convolutional Block
            nn.Conv2d(12, 12, kernel_size=3, bias=False), # 2D Convolution: 12 input channels, 12 output channels, 3x3 kernel
            nn.BatchNorm2d(12),  # Batch Normalization
            nn.ReLU(inplace=True),  # ReLU activation

            nn.MaxPool2d(kernel_size=2),  # Max Pooling: reduces spatial dimensions (e.g., 28x28 -> 14x14)

            # Third Convolutional Block
            nn.Conv2d(12, 12, kernel_size=3, bias=False), # 2D Convolution: 12 input channels, 12 output channels, 3x3 kernel
            nn.BatchNorm2d(12),  # Batch Normalization
            nn.ReLU(inplace=True),  # ReLU activation

            # Fourth Convolutional Block
            nn.Conv2d(12, 12, kernel_size=3, bias=False), # 2D Convolution: 12 input channels, 12 output channels, 3x3 kernel
            nn.BatchNorm2d(12),  # Batch Normalization
            nn.ReLU(inplace=True),  # ReLU activation

            nn.MaxPool2d(kernel_size=2),  # Max Pooling: reduces spatial dimensions again (e.g., 14x14 -> 7x7)

            # Fifth Convolutional Block
            nn.Conv2d(12, 24, kernel_size=3, bias=False), # 2D Convolution: 12 input channels, 24 output channels, 3x3 kernel
            nn.BatchNorm2d(24),  # Batch Normalization
            nn.ReLU(inplace=True),  # ReLU activation

            nn.AdaptiveAvgPool2d(1),  # Global Average Pooling: reduces each feature map to a single value (1x1)

            # Final 1x1 Convolutional Layer for classification
            nn.Conv2d(24, 10, kernel_size=1, bias=False), # 1x1 Convolution: reduces channels to 10 (number of classes)
        )

    def forward(self, x):
        # Define the forward pass of the model.
        # This method describes how the input 'x' is processed through the layers.
        x = self.conv1(x)              # Pass the input through the defined convolutional layers
        # The output of conv1 has a shape of [batch_size, 10, 1, 1] after AdaptiveAvgPool2d and 1x1 Conv
        x = x.view(-1, 10)             # Reshape the tensor to have a size of (batch_size, 10) for classification
        return x                       # Return the final output (logits for 10 classes)