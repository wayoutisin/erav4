"""
Model utilities for ResNet18 training and inference
"""
import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any

from ..models.resnet18 import create_resnet18_model
from .logger_utils import setup_logger


def create_resnet18_model_util(num_classes: int, pretrained: bool = True, **kwargs) -> nn.Module:
    """Create ResNet18 model for classification (utility wrapper)"""
    return create_resnet18_model(num_classes=num_classes, pretrained=pretrained, **kwargs)


def save_model(model: nn.Module, 
               optimizer: torch.optim.Optimizer, 
               epoch: int, 
               accuracy: float, 
               save_path: str, 
               class_names: List[str],
               additional_info: Optional[Dict[str, Any]] = None) -> None:
    """Save model checkpoint with metadata"""
    
    logger = setup_logger('model_utils')
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'accuracy': accuracy,
        'class_names': class_names,
        'num_classes': len(class_names),
        'model_name': 'resnet18'
    }
    
    # Add additional information if provided
    if additional_info:
        checkpoint.update(additional_info)
    
    torch.save(checkpoint, save_path)
    logger.info(f"Model saved to: {save_path}")


def load_model(model_path: str, device: torch.device = None) -> Tuple[nn.Module, List[str]]:
    """Load trained model from checkpoint"""
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger = setup_logger('model_utils')
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create model
    num_classes = checkpoint['num_classes']
    model = create_resnet18_model(num_classes, pretrained=False)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    class_names = checkpoint['class_names']
    
    logger.info(f"Model loaded from: {model_path}")
    logger.info(f"Classes: {class_names}")
    logger.info(f"Accuracy: {checkpoint.get('accuracy', 'N/A')}")
    
    return model, class_names


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Count total and trainable parameters in model"""
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params


def freeze_layers(model: nn.Module, layers_to_freeze: List[str]) -> nn.Module:
    """Freeze specified layers in the model"""
    
    logger = setup_logger('model_utils')
    
    for name, param in model.named_parameters():
        for layer_name in layers_to_freeze:
            if layer_name in name:
                param.requires_grad = False
                logger.info(f"Frozen layer: {name}")
                break
    
    total_params, trainable_params = count_parameters(model)
    logger.info(f"After freezing - Total: {total_params:,}, Trainable: {trainable_params:,}")
    
    return model


def unfreeze_layers(model: nn.Module, layers_to_unfreeze: List[str]) -> nn.Module:
    """Unfreeze specified layers in the model"""
    
    logger = setup_logger('model_utils')
    
    for name, param in model.named_parameters():
        for layer_name in layers_to_unfreeze:
            if layer_name in name:
                param.requires_grad = True
                logger.info(f"Unfrozen layer: {name}")
                break
    
    total_params, trainable_params = count_parameters(model)
    logger.info(f"After unfreezing - Total: {total_params:,}, Trainable: {trainable_params:,}")
    
    return model


def get_model_summary(model: nn.Module) -> Dict[str, Any]:
    """Get comprehensive model summary"""
    
    total_params, trainable_params = count_parameters(model)
    
    summary = {
        'model_name': 'ResNet18',
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'frozen_parameters': total_params - trainable_params,
        'layers': []
    }
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            layer_params = sum(p.numel() for p in module.parameters())
            trainable_layer_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            
            summary['layers'].append({
                'name': name,
                'type': type(module).__name__,
                'parameters': layer_params,
                'trainable_parameters': trainable_layer_params,
                'frozen': layer_params > 0 and trainable_layer_params == 0
            })
    
    return summary


def create_optimizer(model: nn.Module, config: Dict[str, Any]) -> torch.optim.Optimizer:
    """Create optimizer based on configuration"""
    
    optimizer_config = config['training']['optimizer']
    optimizer_name = optimizer_config['name'].lower()
    
    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=optimizer_config['lr'],
            weight_decay=optimizer_config.get('weight_decay', 0),
            betas=optimizer_config.get('betas', (0.9, 0.999))
        )
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=optimizer_config['lr'],
            momentum=optimizer_config.get('momentum', 0.9),
            weight_decay=optimizer_config.get('weight_decay', 0),
            nesterov=optimizer_config.get('nesterov', False)
        )
    elif optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=optimizer_config['lr'],
            weight_decay=optimizer_config.get('weight_decay', 0.01),
            betas=optimizer_config.get('betas', (0.9, 0.999))
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    return optimizer


def create_scheduler(optimizer: torch.optim.Optimizer, config: Dict[str, Any]) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """Create learning rate scheduler based on configuration"""
    
    scheduler_config = config['training'].get('scheduler', {})
    
    if not scheduler_config or 'name' not in scheduler_config:
        return None
    
    scheduler_name = scheduler_config['name'].lower()
    
    if scheduler_name == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_config.get('step_size', 10),
            gamma=scheduler_config.get('gamma', 0.1)
        )
    elif scheduler_name == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=scheduler_config.get('milestones', [30, 60, 90]),
            gamma=scheduler_config.get('gamma', 0.1)
        )
    elif scheduler_name == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_config.get('T_max', config['training']['epochs'])
        )
    elif scheduler_name == 'exponential':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=scheduler_config.get('gamma', 0.95)
        )
    elif scheduler_name == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=scheduler_config.get('mode', 'min'),
            factor=scheduler_config.get('factor', 0.1),
            patience=scheduler_config.get('patience', 10),
            min_lr=scheduler_config.get('min_lr', 0)
        )
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")
    
    return scheduler


def apply_weight_init(model: nn.Module, init_type: str = 'xavier') -> nn.Module:
    """Apply weight initialization to model"""
    
    logger = setup_logger('model_utils')
    
    def init_weights(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            if init_type == 'xavier':
                nn.init.xavier_uniform_(m.weight)
            elif init_type == 'kaiming':
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            elif init_type == 'normal':
                nn.init.normal_(m.weight, 0, 0.02)
            
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    model.apply(init_weights)
    logger.info(f"Applied {init_type} weight initialization")
    
    return model


def export_model_to_onnx(model: nn.Module, 
                        save_path: str, 
                        input_shape: Tuple[int, ...] = (1, 3, 224, 224),
                        device: torch.device = None) -> None:
    """Export model to ONNX format"""
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger = setup_logger('model_utils')
    
    model.eval()
    dummy_input = torch.randn(input_shape).to(device)
    
    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    logger.info(f"Model exported to ONNX: {save_path}")


def calculate_model_flops(model: nn.Module, input_shape: Tuple[int, ...] = (1, 3, 224, 224)) -> int:
    """Calculate FLOPs for the model (requires thop package)"""
    
    try:
        from thop import profile
        
        model.eval()
        dummy_input = torch.randn(input_shape)
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        
        return flops, params
    except ImportError:
        logger = setup_logger('model_utils')
        logger.warning("thop package not found. Install with 'pip install thop' to calculate FLOPs")
        return None, None