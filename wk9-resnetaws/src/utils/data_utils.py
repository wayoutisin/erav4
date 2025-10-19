"""
Data loading utilities for ResNet18 training
"""
import os
from pathlib import Path
from typing import Tuple, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split

from .logger_utils import setup_logger


def get_data_transforms(config: dict) -> Tuple[transforms.Compose, transforms.Compose]:
    """Get training and validation transforms"""
    
    image_size = config['data']['image_size']
    augmentation_config = config['data'].get('augmentation', {})
    
    # Training transforms with augmentation
    train_transform_list = [
        transforms.Resize(256),
        transforms.RandomResizedCrop(image_size)
    ]
    
    # Add augmentations if specified
    if augmentation_config.get('random_horizontal_flip', False):
        train_transform_list.append(transforms.RandomHorizontalFlip())
    
    if augmentation_config.get('random_rotation', False):
        rotation_degrees = augmentation_config['random_rotation']
        train_transform_list.append(transforms.RandomRotation(rotation_degrees))
    
    if 'color_jitter' in augmentation_config:
        jitter_params = augmentation_config['color_jitter']
        train_transform_list.append(transforms.ColorJitter(
            brightness=jitter_params.get('brightness', 0),
            contrast=jitter_params.get('contrast', 0),
            saturation=jitter_params.get('saturation', 0),
            hue=jitter_params.get('hue', 0)
        ))
    
    # Add normalization
    train_transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_transform = transforms.Compose(train_transform_list)
    
    # Validation transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


def create_data_loaders_from_folder(data_path: str, 
                                   config: dict, 
                                   train_transform: transforms.Compose,
                                   val_transform: transforms.Compose) -> Tuple[DataLoader, DataLoader, List[str]]:
    """Create data loaders from folder structure"""
    
    data_path = Path(data_path)
    logger = setup_logger('data_loader')
    
    train_split = config['data'].get('train_split', 0.8)
    batch_size = config['data']['batch_size']
    num_workers = config['data']['num_workers']
    
    # Check if train/val folders exist
    train_path = data_path / 'train'
    val_path = data_path / 'val'
    
    if train_path.exists() and val_path.exists():
        logger.info("Found separate train/val folders")
        
        train_dataset = ImageFolder(train_path, transform=train_transform)
        val_dataset = ImageFolder(val_path, transform=val_transform)
        class_names = train_dataset.classes
        
    else:
        logger.info("Single folder found, splitting dataset")
        
        # Load full dataset
        full_dataset = ImageFolder(data_path, transform=train_transform)
        class_names = full_dataset.classes
        
        # Split dataset
        train_size = int(train_split * len(full_dataset))
        val_size = len(full_dataset) - train_size
        
        train_dataset, val_dataset = random_split(
            full_dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(config.get('seed', 42))
        )
        
        # Apply different transforms to validation set
        val_dataset.dataset.transform = val_transform
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    logger.info(f"Created data loaders with {len(train_dataset)} train and {len(val_dataset)} val samples")
    
    return train_loader, val_loader, class_names


def get_data_loaders(config: dict) -> Tuple[DataLoader, DataLoader, List[str]]:
    """Main function to get data loaders based on configuration"""
    
    logger = setup_logger('data_loader')
    
    # Get transforms
    train_transform, val_transform = get_data_transforms(config)
    
    dataset_path = config['data']['dataset_path']
    
    if os.path.exists(dataset_path):
        logger.info(f"Loading data from: {dataset_path}")
        return create_data_loaders_from_folder(
            dataset_path, config, train_transform, val_transform
        )
    else:
        logger.warning(f"Dataset path not found: {dataset_path}")
        logger.info("Using CIFAR-10 as fallback dataset")
        return create_cifar10_loaders(config, train_transform, val_transform)


def create_cifar10_loaders(config: dict, 
                          train_transform: transforms.Compose,
                          val_transform: transforms.Compose) -> Tuple[DataLoader, DataLoader, List[str]]:
    """Create CIFAR-10 data loaders as fallback"""
    
    from torchvision.datasets import CIFAR10
    
    batch_size = config['data']['batch_size']
    num_workers = config['data']['num_workers']
    
    # Download and create datasets
    train_dataset = CIFAR10(
        root='data/cifar10', 
        train=True, 
        download=True, 
        transform=train_transform
    )
    
    val_dataset = CIFAR10(
        root='data/cifar10', 
        train=False, 
        download=True, 
        transform=val_transform
    )
    
    # CIFAR-10 class names
    class_names = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader, class_names


class CustomImageDataset(Dataset):
    """Custom dataset class for flexible data loading"""
    
    def __init__(self, image_paths: List[str], labels: List[int], transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        from PIL import Image
        
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def create_custom_data_loaders(image_paths: List[str],
                              labels: List[int],
                              class_names: List[str],
                              config: dict,
                              test_size: float = 0.2) -> Tuple[DataLoader, DataLoader, List[str]]:
    """Create data loaders from lists of image paths and labels"""
    
    train_transform, val_transform = get_data_transforms(config)
    
    # Split data
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, 
        test_size=test_size, 
        random_state=config.get('seed', 42),
        stratify=labels
    )
    
    # Create datasets
    train_dataset = CustomImageDataset(train_paths, train_labels, train_transform)
    val_dataset = CustomImageDataset(val_paths, val_labels, val_transform)
    
    # Create data loaders
    batch_size = config['data']['batch_size']
    num_workers = config['data']['num_workers']
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader, class_names