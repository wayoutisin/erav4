# Import necessary libraries for building the neural network
import torch  # PyTorch library for tensor operations and neural networks
import torch.nn as nn  # Neural network module for defining layers

class MNISTModelLL(nn.Module):
    """Your model with configurable dropout rates"""
    # This class defines the neural network model for MNIST digit classification.
    # It inherits from nn.Module, which is the base class for all neural network modules in PyTorch.
    def __init__(self):
        super(MNISTModelLL, self).__init__()

        # Define the convolutional layers using nn.Sequential
        # nn.Sequential is a container that holds a sequence of modules.
        # The input will be passed through all modules in the same order as they are added.
        self.conv1 = nn.Sequential(
            # First convolutional block
            nn.Conv2d(1, 8, kernel_size=3, bias=False),  # 2D Convolution layer: 1 input channel (grayscale image), 8 output channels, 3x3 kernel
            nn.BatchNorm2d(8),  # Batch Normalization: normalizes the output of the previous layer
            nn.ReLU(inplace=True),  # ReLU activation function: introduces non-linearity
            nn.Dropout(0.05),  # Dropout layer: randomly sets a fraction of input units to 0 to prevent overfitting

            # Second convolutional block
            nn.Conv2d(8, 8, kernel_size=3, bias=False),  # 2D Convolution layer: 8 input channels, 8 output channels, 3x3 kernel
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Dropout(0.05),

            nn.MaxPool2d(kernel_size=2),  # Max Pooling layer: reduces the spatial dimensions (e.g., 28x28 -> 14x14)

            # Third convolutional block
            nn.Conv2d(8, 16, kernel_size=3, bias=False), # 2D Convolution layer: 8 input channels, 16 output channels, 3x3 kernel
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Dropout(0.05),


            # Fourth convolutional block
            nn.Conv2d(16, 16, kernel_size=3, bias=False), # 2D Convolution layer: 16 input channels, 16 output channels, 3x3 kernel
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Dropout(0.05),

            nn.MaxPool2d(kernel_size=2), # Max Pooling layer: reduces spatial dimensions again

            # Fifth convolutional block
            nn.Conv2d(16, 32, kernel_size=3, bias=False), # 2D Convolution layer: 16 input channels, 32 output channels, 3x3 kernel
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.05),

            # Sixth convolutional layer (1x1 convolution for channel reduction)
            nn.Conv2d(32, 10, kernel_size=1, bias=False), # 1x1 Convolution: used to reduce channels to 10 (number of classes)
            nn.BatchNorm2d(10),
            nn.ReLU(inplace=True),

        )

        # Global Average Pooling and Linear layers (Fully Connected)
        # This section processes the output from the convolutional layers to produce final class scores.
        self.gap_and_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global Average Pooling: reduces each feature map to a single value (1x1)
            nn.Flatten(),  # Flattens the output into a 1D tensor
            nn.Linear(10, 10, bias=False),  # First Linear layer: maps 10 input features to 10 output features (number of classes)
            nn.ReLU(),  # ReLU activation
            nn.Linear(10, 10, bias=False)  # Second Linear layer: final output layer with 10 classes
        )

    def forward(self, x):
        # Define the forward pass of the model.
        # This method describes how the input 'x' is processed through the layers.
        x = self.conv1(x)              # Pass the input through the convolutional layers
        x = self.gap_and_fc(x)         # Pass the output of conv layers through Global Average Pooling and Linear layers
        return x                       # Return the final output (logits for 10 classes)