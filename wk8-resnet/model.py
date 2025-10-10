"""
ResNet model definition for CIFAR-100, adapted from notebook implementation.
Includes ResNet-56 and ResNet-110 architectures with BasicBlock for CIFAR inputs.
"""
import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    """
    Basic residual block for ResNet (CIFAR version).
    Implements two 3x3 convolutions with BatchNorm and ReLU, plus a shortcut connection.
    Handles downsampling and channel increase using a 1x1 convolution if needed.
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # Shortcut is either identity or 1x1 conv for shape match
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(identity)
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    """
    ResNet architecture for CIFAR datasets (e.g., CIFAR-10 or CIFAR-100).
    - Uses three stages/"layers" with increasing feature depths.
    - Designed for small images (32x32), so it uses 3x3 convolutions.
    - Each layer consists of multiple BasicBlocks, with downsampling in layer2 and layer3.
    """
    def __init__(self, block, num_blocks, num_classes=100):
        super(ResNet, self).__init__()
        self.in_channels = 16
        # Initial convolutional layer (3x3)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        # Residual layers
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        # Global average pooling and final linear classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)
        # Weight initialization: Kaiming fan_out for conv and linear
        self._initialize_weights()

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """
        Helper to create a stack of BasicBlocks for a given stage.
        First block of the layer may do downsampling.
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

def ResNet56(num_classes=100):
    """
    Constructs a ResNet-56 (3x9=27 blocks; 56 layers) for CIFAR-100/10 input.
    """
    return ResNet(BasicBlock, [9, 9, 9], num_classes=num_classes)

def ResNet110(num_classes=100):
    """
    Constructs a ResNet-110 (3x18=54 blocks; 110 layers) for CIFAR input.
    """
    return ResNet(BasicBlock, [18, 18, 18], num_classes=num_classes)
