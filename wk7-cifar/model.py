import torch.nn as nn
import torch.nn.functional as F

# Basic CNN Block implementing Depthwise Separable Convolution with Receptive Field = 7
class BasicBlock(nn.Module):
    """
    A custom CNN block that combines standard convolution with depthwise separable convolution.
    
    This block achieves a receptive field of 7 using 4 convolutional layers:
    1. Standard 3x3 convolution
    2. Depthwise separable convolution (depthwise + pointwise)
    3. Standard 3x3 convolution (with optional stride for downsampling)
    
    Receptive Field Calculation:
    - conv1 (3x3): RF = 3
    - conv2 (3x3 depthwise): RF = 3 + (3-1) = 5  
    - conv3 (1x1 pointwise): RF = 5 + (1-1) = 5
    - conv4 (3x3): RF = 5 + (3-1) = 7
    
    Total Receptive Field = 7 pixels
    """
    
    def __init__(self, in_channels, out_channels, stride=1):
        """
        Initialize the BasicBlock with specified channels and stride.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels  
            stride: Stride for the final conv layer (used for downsampling)
        """
        super(BasicBlock, self).__init__()
        
        # First conv layer: Standard 3x3 convolution for initial feature extraction
        # RF contribution: 3 pixels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)  # Batch normalization for stability
        self.do1 = nn.Dropout(0.05)  # Light dropout for regularization
        
        # Depthwise separable convolution: More efficient than standard conv
        # Step 1: Depthwise convolution (each input channel processed separately)
        # RF contribution: (3-1) = 2 pixels
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False, groups=out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.do2 = nn.Dropout(0.05)

        # Step 2: Pointwise convolution (1x1 conv to mix channels)
        # RF contribution: (1-1) = 0 pixels (no spatial expansion)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1,
                              stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.do3 = nn.Dropout(0.05)
     
        # Final conv layer: Standard 3x3 convolution with optional stride
        # RF contribution: (3-1) = 2 pixels
        # This layer can downsample the spatial dimensions when stride > 1
        self.conv4 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=stride, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(out_channels)
        
    
    def forward(self, x):
        """
        Forward pass through the BasicBlock.
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)
            
        Returns:
            Output tensor of shape (batch_size, out_channels, height/stride, width/stride)
        """
        # Layer 1: Standard 3x3 convolution with ReLU activation and dropout
        out = F.relu(self.bn1(self.conv1(x)))  # Apply batch norm then ReLU
        out = self.do1(out)  # Apply dropout for regularization
        
        # Layer 2: Depthwise convolution (processes each channel separately)
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.do2(out)
        
        # Layer 3: Pointwise convolution (1x1 conv to mix channels)
        out = F.relu(self.bn3(self.conv3(out)))
        out = self.do3(out)
        
        # Layer 4: Final 3x3 convolution (may downsample if stride > 1)
        out = F.relu(self.bn4(self.conv4(out)))

        return out

# Final CNN Architecture optimized for CIFAR-10 with <200k parameters
class FinalCNN(nn.Module):
    """
    Optimized CNN architecture for CIFAR-10 classification.
    
    This model uses a progressive channel expansion strategy with depthwise separable
    convolutions to achieve good performance while keeping parameters under 200k.
    
    Architecture Overview:
    - Input: 32x32x3 (CIFAR-10 images)
    - Progressive channel expansion: 3 -> 12 -> 24 -> 36 -> 72
    - Total receptive field: 7 pixels per block
    """
    
    def __init__(self, num_classes=10):
        """
        Initialize the FinalCNN model.
        
        Args:
            num_classes: Number of output classes (10 for CIFAR-10)
        """
        super(FinalCNN, self).__init__()
        
        # Initial convolution layer: Convert RGB to initial feature maps
        # Input: 32x32x3, Output: 32x32x12
        # RF: 3 pixels
        self.conv1 = nn.Conv2d(3, 12, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(12)
        
        # Block 1: Maintain spatial resolution, double channels
        # Input: 32x32x12, Output: 32x32x24
        # RF: 3 + 7 = 10 pixels
        self.block1 = BasicBlock(12, 24, stride=1)
        
        # Block 2: First downsampling, maintain channels
        # Input: 32x32x24, Output: 16x16x24 (stride=2 downsamples)
        # RF: 10 + 7 = 17 pixels
        self.block2 = BasicBlock(24, 24, stride=2)
        
        # Block 3: Increase channels, maintain spatial resolution
        # Input: 16x16x24, Output: 16x16x36
        # RF: 17 + 7 = 24 pixels
        self.block3 = BasicBlock(24, 36, stride=1)
        
        # Block 4: Second downsampling, maintain channels
        # Input: 16x16x36, Output: 8x8x36 (stride=2 downsamples)
        # RF: 24 + 7 = 31 pixels
        self.block4 = BasicBlock(36, 36, stride=2)
        
        # Block 5: Final channel expansion, maintain spatial resolution
        # Input: 8x8x36, Output: 8x8x72
        # RF: 31 + 7 = 38 pixels
        self.block5 = BasicBlock(36, 72, stride=1)
        
        # Global Average Pooling: Convert spatial features to single values
        # Input: 8x8x72, Output: 1x1x72
        # This replaces fully connected layers and reduces overfitting
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Final classifier: Convert features to class predictions
        # Input: 72 features, Output: 10 classes
        self.fc = nn.Linear(72, num_classes)
        
        
    
    def forward(self, x):
        """
        Forward pass through the entire CNN architecture.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 32, 32)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Initial convolution: Convert RGB to feature maps
        # 32x32x3 -> 32x32x12
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Block 1: Feature extraction at full resolution
        # 32x32x12 -> 32x32x24, RF = 10 pixels
        x = self.block1(x)
        
        # Block 2: First downsampling for efficiency
        # 32x32x24 -> 16x16x24, RF = 17 pixels
        x = self.block2(x)
        
        # Block 3: Channel expansion at reduced resolution
        # 16x16x24 -> 16x16x36, RF = 24 pixels
        x = self.block3(x)
        
        # Block 4: Second downsampling for deeper features
        # 16x16x36 -> 8x8x36, RF = 31 pixels
        x = self.block4(x)
        
        # Block 5: Final channel expansion
        # 8x8x36 -> 8x8x72, RF = 38 pixels
        x = self.block5(x)
        
        # Global Average Pooling: Convert spatial features to global descriptors
        # 8x8x72 -> 1x1x72
        x = self.global_avg_pool(x)
        
        # Flatten for classification
        # 1x1x72 -> 72
        x = x.view(x.size(0), -1)
        
        # Final classification layer
        # 72 -> 10 (num_classes)
        x = self.fc(x)
        
        return x
