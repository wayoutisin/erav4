#!/usr/bin/env python3
"""
ResNet18 Training Script for Image Classification
"""
import os
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from tqdm import tqdm

from ..models.resnet18 import create_resnet18_model, print_model_architecture
from ..utils.data_utils import get_data_loaders
from ..utils.model_utils import save_model, load_model
from ..utils.logger_utils import setup_logger
from ..utils.config_utils import SecureConfigLoader


class ResNet18Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.logger = setup_logger('trainer', config.get('log_level', 'INFO'))
        
        # Create output directories
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir = self.output_dir / 'models'
        self.model_dir.mkdir(exist_ok=True)
        
        self.logger.info(f"Using device: {self.device}")
        
    def build_model(self):
        """Build ResNet18 model for classification"""
        num_classes = self.config['model']['num_classes']
        pretrained = self.config['model'].get('pretrained', True)
        use_custom_head = self.config['model'].get('custom_head', False)
        dropout = self.config['model'].get('dropout', 0.0)
        use_auxiliary = self.config['model'].get('auxiliary_classifier', False)
        
        # Create model using our factory function
        self.model = create_resnet18_model(
            num_classes=num_classes,
            pretrained=pretrained,
            custom_head=use_custom_head,
            dropout=dropout,
            use_auxiliary=use_auxiliary
        )
        
        self.model = self.model.to(self.device)
        
        # Print model architecture
        if self.config.get('print_architecture', False):
            print_model_architecture(self.model)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        optimizer_config = self.config['training']['optimizer']
        if optimizer_config['name'] == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=optimizer_config['lr'],
                weight_decay=optimizer_config.get('weight_decay', 0)
            )
        elif optimizer_config['name'] == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=optimizer_config['lr'],
                momentum=optimizer_config.get('momentum', 0.9),
                weight_decay=optimizer_config.get('weight_decay', 0)
            )
        
        # Learning rate scheduler
        scheduler_config = self.config['training'].get('scheduler', {})
        if scheduler_config.get('name') == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config.get('step_size', 10),
                gamma=scheduler_config.get('gamma', 0.1)
            )
        elif scheduler_config.get('name') == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['epochs']
            )
        
        self.logger.info(f"Model built with {num_classes} classes")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        predictions = []
        targets = []
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}')
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            predictions.extend(pred.cpu().numpy().flatten())
            targets.extend(target.cpu().numpy())
            
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = accuracy_score(targets, predictions)
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, val_loader):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        predictions = []
        targets = []
        
        with torch.no_grad():
            for data, target in tqdm(val_loader, desc='Validation'):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                running_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                predictions.extend(pred.cpu().numpy().flatten())
                targets.extend(target.cpu().numpy())
        
        val_loss = running_loss / len(val_loader)
        val_acc = accuracy_score(targets, predictions)
        
        return val_loss, val_acc, predictions, targets
    
    def train(self):
        """Main training loop"""
        # Get data loaders
        train_loader, val_loader, class_names = get_data_loaders(self.config)
        
        # Build model
        self.build_model()
        
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        best_val_acc = 0.0
        best_model_path = None
        
        self.logger.info("Starting training...")
        
        for epoch in range(self.config['training']['epochs']):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_loss, val_acc, predictions, targets = self.validate_epoch(val_loader)
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
            
            # Save history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Log results
            self.logger.info(
                f"Epoch {epoch+1}/{self.config['training']['epochs']} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_path = self.model_dir / f'best_model_epoch_{epoch+1}.pth'
                save_model(
                    self.model, 
                    self.optimizer, 
                    epoch, 
                    val_acc, 
                    best_model_path,
                    class_names
                )
                self.logger.info(f"New best model saved with validation accuracy: {val_acc:.4f}")
        
        # Save final model
        final_model_path = self.model_dir / 'final_model.pth'
        save_model(
            self.model, 
            self.optimizer, 
            epoch, 
            val_acc, 
            final_model_path,
            class_names
        )
        
        # Save training history
        history_path = self.output_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        # Generate final classification report
        report = classification_report(targets, predictions, target_names=class_names)
        self.logger.info(f"Final Classification Report:\n{report}")
        
        # Save classification report
        report_path = self.output_dir / 'classification_report.txt'
        with open(report_path, 'w') as f:
            f.write(report)
        
        self.logger.info(f"Training completed. Best validation accuracy: {best_val_acc:.4f}")
        self.logger.info(f"Models saved in: {self.model_dir}")
        
        return history, best_model_path, final_model_path


def main():
    parser = argparse.ArgumentParser(description='Train ResNet18 for image classification')
    parser.add_argument('--config', type=str, help='Path to config file (optional - will use secure loader)')
    parser.add_argument('--output-dir', type=str, help='Override output directory')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, help='Batch size for training')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--local-test', action='store_true', help='Run quick local test (5 epochs, small batch)')
    parser.add_argument('--from-scratch', action='store_true', help='Train from scratch (no pretrained weights)')
    
    args = parser.parse_args()
    
    # Load configuration securely
    if args.config:
        # Traditional config file loading
        with open(args.config, 'r') as f:
            config = json.load(f)
        print(f"Loaded configuration from {args.config}")
    elif args.local_test:
        # Use local test configuration
        config_path = Path(__file__).parent.parent.parent / 'configs' / 'local_test_config.json'
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"Loaded local test configuration from {config_path}")
    else:
        # Secure configuration loading
        try:
            config_loader = SecureConfigLoader()
            config = config_loader.load_training_config()
            print("Loaded configuration securely from environment variables")
        except Exception as e:
            print(f"Failed to load secure configuration: {e}")
            print("Either provide --config argument or set up environment variables")
            config_loader.print_required_env_vars()
            return
    
    # Apply command line overrides
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['data']['batch_size'] = args.batch_size
    if args.lr:
        config['training']['optimizer']['lr'] = args.lr
    if args.output_dir:
        config['output']['output_dir'] = args.output_dir
    if args.from_scratch or args.local_test:
        config['model']['pretrained'] = False
        if 'training' in config and 'pretrained' in config['training']:
            config['training']['pretrained'] = False
        print("Training from scratch (no pretrained weights)")
    
    # Local test specific settings
    if args.local_test:
        config['training']['epochs'] = min(config['training']['epochs'], 5)
        config['data']['batch_size'] = min(config['data']['batch_size'], 16)
        config['output']['output_dir'] = 'outputs/local_test'
        print("Local test mode: 5 epochs max, batch size 16 max")
    
    print(f"Training configuration:")
    print(f"  Epochs: {config['training']['epochs']}")
    print(f"  Batch size: {config['data']['batch_size']}")
    print(f"  Learning rate: {config['training']['optimizer']['lr']}")
    print(f"  Pretrained: {config['model']['pretrained']}")
    print(f"  Output directory: {config['output']['output_dir']}")
    
    # Create trainer and start training
    trainer = ResNet18Trainer(config)
    history, best_model_path, final_model_path = trainer.train()
    
    print(f"Training completed successfully!")
    print(f"Best model: {best_model_path}")
    print(f"Final model: {final_model_path}")
    
    # Print summary statistics
    if history:
        best_val_acc = max(history['val_accuracy']) if 'val_accuracy' in history else 0
        final_train_loss = history['train_loss'][-1] if 'train_loss' in history else 0
        final_val_loss = history['val_loss'][-1] if 'val_loss' in history else 0
        
        print(f"\nTraining Summary:")
        print(f"  Best validation accuracy: {best_val_acc:.4f}")
        print(f"  Final training loss: {final_train_loss:.4f}")
        print(f"  Final validation loss: {final_val_loss:.4f}")
        
        if args.local_test:
            print(f"\n✓ Local test completed successfully!")
            print(f"  Ready for full training or AWS deployment")
        elif not config['model']['pretrained']:
            print(f"\n✓ Training from scratch completed!")
            print(f"  Model learned without pretrained weights")


if __name__ == '__main__':
    main()