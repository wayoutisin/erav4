"""
ResNet18 model implementation from scratch
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class BasicBlock(nn.Module):
    """Basic building block for ResNet18/34"""
    expansion = 1
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, 
                 downsample: Optional[nn.Module] = None):
        super(BasicBlock, self).__init__()
        
        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Second convolutional layer
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Downsample for skip connection if needed
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        # First conv block
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        
        # Second conv block
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Skip connection
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = F.relu(out, inplace=True)
        
        return out


class ResNet18(nn.Module):
    """ResNet18 architecture implementation"""
    
    def __init__(self, num_classes: int = 1000, zero_init_residual: bool = False):
        super(ResNet18, self).__init__()
        
        self.num_classes = num_classes
        self.in_channels = 64
        
        # Initial convolutional layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet layers
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        
        # Final layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)
        
        # Initialize weights
        self._initialize_weights(zero_init_residual)
    
    def _make_layer(self, out_channels: int, blocks: int, stride: int = 1) -> nn.Sequential:
        """Create a layer with multiple basic blocks"""
        downsample = None
        
        # Create downsample layer if needed
        if stride != 1 or self.in_channels != out_channels * BasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * BasicBlock.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion),
            )
        
        layers = []
        layers.append(BasicBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * BasicBlock.expansion
        
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self, zero_init_residual: bool = False):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # Zero-initialize the last BN in each residual branch
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial conv layer
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.maxpool(x)
        
        # ResNet layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Final layers
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x
    
    def get_feature_maps(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Get intermediate feature maps for visualization"""
        features = []
        
        # Initial conv
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        features.append(x.clone())
        x = self.maxpool(x)
        
        # ResNet layers
        x = self.layer1(x)
        features.append(x.clone())
        
        x = self.layer2(x)
        features.append(x.clone())
        
        x = self.layer3(x)
        features.append(x.clone())
        
        x = self.layer4(x)
        features.append(x.clone())
        
        return features
    
    def get_model_info(self) -> dict:
        """Get model architecture information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'name': 'ResNet18',
            'num_classes': self.num_classes,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'input_size': (3, 224, 224),
            'layers': {
                'conv1': '7x7, 64, stride=2',
                'layer1': '2 x [3x3, 64]',
                'layer2': '2 x [3x3, 128], stride=2',
                'layer3': '2 x [3x3, 256], stride=2',
                'layer4': '2 x [3x3, 512], stride=2',
                'avgpool': 'AdaptiveAvgPool2d(1,1)',
                'fc': f'Linear(512, {self.num_classes})'
            }
        }


class ResNet18Classifier(nn.Module):
    """ResNet18 with additional classification utilities"""
    
    def __init__(self, num_classes: int, pretrained: bool = False, 
                 dropout: float = 0.0, use_auxiliary: bool = False):
        super(ResNet18Classifier, self).__init__()
        
        if pretrained:
            # Load pretrained torchvision model and modify
            from torchvision import models
            self.backbone = models.resnet18(pretrained=True)
            # Remove the final layer
            self.backbone.fc = nn.Identity()
            self.feature_dim = 512
        else:
            # Use our custom implementation
            self.backbone = ResNet18(num_classes=1000)  # Initialize with 1000, we'll replace fc
            self.backbone.fc = nn.Identity()
            self.feature_dim = 512
        
        # Custom classifier head
        classifier_layers = []
        
        if dropout > 0:
            classifier_layers.append(nn.Dropout(dropout))
        
        classifier_layers.append(nn.Linear(self.feature_dim, num_classes))
        
        self.classifier = nn.Sequential(*classifier_layers)
        
        # Auxiliary classifier for better gradient flow (optional)
        self.use_auxiliary = use_auxiliary
        if use_auxiliary:
            self.aux_classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(256, num_classes)  # From layer3 output
            )
        
        self.num_classes = num_classes
    
    def forward(self, x: torch.Tensor, return_features: bool = False):
        # Extract features
        features = self.extract_features(x)
        
        # Main classification
        logits = self.classifier(features)
        
        if self.training and self.use_auxiliary:
            # Auxiliary classification from layer3
            layer3_out = self._get_layer3_output(x)
            aux_logits = self.aux_classifier(layer3_out)
            
            if return_features:
                return logits, aux_logits, features
            return logits, aux_logits
        
        if return_features:
            return logits, features
        
        return logits
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from backbone"""
        return self.backbone(x)
    
    def _get_layer3_output(self, x: torch.Tensor) -> torch.Tensor:
        """Get layer3 output for auxiliary classifier"""
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        
        return x
    
    def freeze_backbone(self, freeze: bool = True):
        """Freeze/unfreeze backbone parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = not freeze
    
    def unfreeze_layers(self, layers: List[str]):
        """Unfreeze specific layers"""
        for name, param in self.backbone.named_parameters():
            for layer_name in layers:
                if layer_name in name:
                    param.requires_grad = True
                    break


def create_resnet18_model(num_classes: int, pretrained: bool = True, 
                         custom_head: bool = False, **kwargs) -> nn.Module:
    """Factory function to create ResNet18 model"""
    
    if custom_head:
        # Use our custom classifier with additional options
        model = ResNet18Classifier(
            num_classes=num_classes, 
            pretrained=pretrained,
            dropout=kwargs.get('dropout', 0.0),
            use_auxiliary=kwargs.get('use_auxiliary', False)
        )
    else:
        if pretrained:
            # Use torchvision pretrained model
            from torchvision import models
            model = models.resnet18(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        else:
            # Use our custom implementation
            model = ResNet18(num_classes=num_classes)
    
    return model


def print_model_architecture(model: nn.Module, input_size: tuple = (1, 3, 224, 224)):
    """Print detailed model architecture"""
    print("=" * 80)
    print("MODEL ARCHITECTURE")
    print("=" * 80)
    
    if hasattr(model, 'get_model_info'):
        info = model.get_model_info()
        print(f"Model: {info['name']}")
        print(f"Classes: {info['num_classes']}")
        print(f"Parameters: {info['total_parameters']:,}")
        print(f"Input Size: {info['input_size']}")
        print("\nLayer Details:")
        for layer, desc in info['layers'].items():
            print(f"  {layer}: {desc}")
    else:
        print(f"Model: {model.__class__.__name__}")
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total Parameters: {total_params:,}")
    
    print("\n" + "=" * 80)
    print("DETAILED ARCHITECTURE")
    print("=" * 80)
    print(model)
    
    # Try to show model summary with input
    try:
        from torchsummary import summary
        if torch.cuda.is_available():
            model = model.cuda()
        summary(model, input_size[1:])
    except ImportError:
        print("\nInstall torchsummary for detailed layer information:")
        print("pip install torchsummary")
    except Exception as e:
        print(f"\nCould not generate summary: {e}")


if __name__ == '__main__':
    # Test the model
    print("Testing ResNet18 implementations...")
    
    # Test custom ResNet18
    print("\n1. Custom ResNet18:")
    model1 = ResNet18(num_classes=10)
    print_model_architecture(model1)
    
    # Test custom classifier
    print("\n2. Custom ResNet18 Classifier:")
    model2 = ResNet18Classifier(num_classes=10, pretrained=False, dropout=0.2)
    print_model_architecture(model2)
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    
    with torch.no_grad():
        out1 = model1(x)
        out2 = model2(x)
        
        print(f"\nOutput shapes:")
        print(f"Custom ResNet18: {out1.shape}")
        print(f"Custom Classifier: {out2.shape}")
        
        # Test feature extraction
        features = model1.get_feature_maps(x)
        print(f"\nFeature map shapes:")
        for i, feat in enumerate(features):
            print(f"  Layer {i+1}: {feat.shape}")
    
    print("\n✅ All tests passed!")