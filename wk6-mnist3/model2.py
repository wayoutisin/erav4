# Import necessary libraries for building the neural network
import torch  # PyTorch library for tensor operations and neural networks
import torch.nn as nn  # Neural network module for defining layers

class MNISTModelGAP(nn.Module):
    """Your model with configurable dropout rates"""
    # This class defines a convolutional neural network model for MNIST digit classification.
    # It inherits from nn.Module, the base class for all neural network modules in PyTorch.
    def __init__(self):
        super(MNISTModelGAP, self).__init__()

        # Define the convolutional layers using nn.Sequential
        # nn.Sequential is a container that holds a sequence of modules.
        # The input will be passed through all modules in the same order as they are added.
        self.conv1 = nn.Sequential(
            # First convolutional block
            nn.Conv2d(1, 12, kernel_size=3, bias=False),  # 2D Convolution layer: 1 input channel (grayscale), 12 output channels, 3x3 kernel
            nn.BatchNorm2d(12),  # Batch Normalization: normalizes the output of the previous layer
            nn.ReLU(inplace=True),  # ReLU activation function: introduces non-linearity
            nn.Dropout(0.05),  # Dropout layer: randomly sets a fraction of input units to 0 to prevent overfitting

            # Second convolutional block
            nn.Conv2d(12, 12, kernel_size=3, bias=False), # 2D Convolution layer: 12 input channels, 12 output channels, 3x3 kernel
            nn.BatchNorm2d(12),
            nn.ReLU(inplace=True),
            nn.Dropout(0.05),

            nn.MaxPool2d(kernel_size=2),  # Max Pooling layer: reduces the spatial dimensions (e.g., 28x28 -> 14x14)

            # Third convolutional block
            nn.Conv2d(12, 12, kernel_size=3, bias=False), # 2D Convolution layer: 12 input channels, 12 output channels, 3x3 kernel
            nn.BatchNorm2d(12),
            nn.ReLU(inplace=True),
            nn.Dropout(0.05),


            # Fourth convolutional block
            nn.Conv2d(12, 12, kernel_size=3, bias=False), # 2D Convolution layer: 12 input channels, 12 output channels, 3x3 kernel
            nn.BatchNorm2d(12),
            nn.ReLU(inplace=True),
            nn.Dropout(0.05),

            nn.MaxPool2d(kernel_size=2), # Max Pooling layer: reduces spatial dimensions again

            # Fifth convolutional block
            nn.Conv2d(12, 24, kernel_size=3, bias=False), # 2D Convolution layer: 12 input channels, 24 output channels, 3x3 kernel
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.Dropout(0.05),

            # Sixth convolutional layer (1x1 convolution for channel reduction)
            nn.Conv2d(24, 10, kernel_size=1, bias=False), # 1x1 Convolution: used to reduce channels to 10 (number of classes)
            nn.BatchNorm2d(10),
            nn.ReLU(inplace=True),

        )

        # Global Average Pooling and (commented out) Linear layers
        # This section processes the output from the convolutional layers.
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1)  # Global Average Pooling: reduces each feature map to a single value (1x1)
        )


    def forward(self, x):
        # Define the forward pass of the model.
        # This method describes how the input 'x' is processed through the layers.
        x = self.conv1(x)              # Pass the input through the convolutional layers
        x = self.gap(x)         # Pass the output of conv layers through Global Average Pooling and Linear layers
        x = x.view(-1, 10)             # Reshape the tensor to have a size of (batch_size, 10) for classification
        return x                       # Return the final output (logits for 10 classes)