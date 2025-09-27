import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchsummary import summary
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from torch.optim.lr_scheduler import StepLR
import argparse  # Import the argparse library for command-line argument parsing
from model1 import MNISTModelLL  # Import the MNISTModelLL from model.py
from model2 import MNISTModelGAP
from model3 import MNISTModelFinal
import logging # Import the logging module
import os # Import the os module for interacting with the operating system

def load_mnist_data(batch_size=64):
    """Load and preprocess MNIST dataset with affine translate augmentation"""
    # Data transformations for training with augmentation
    transform_train = transforms.Compose([
        transforms.RandomRotation((-7.0, 7.0), fill=(1,)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Download and load datasets
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train model for one epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc='Training')
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(data)
        loss = criterion(output, target)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)

        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{100. * correct / total:.2f}%'
        })

    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

def test_epoch(model, test_loader, criterion, device):
    """Test model"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Testing')
        for data, target in pbar:
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100. * correct / total:.2f}%'
            })

    avg_loss = total_loss / len(test_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

def count_parameters(model):
    """Count trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main(model):
    # Training configuration
    config = {
        'batch_size': 32,
        'epochs': 15,
        'lr': 0.001,
        'weight_decay': 0.0001,
        'scheduler': 'plateau',
        'patience': 3,
        'factor': 0.5,
        'min_lr': 1e-6,
        'dropout': (0.05, 0.05, 0.05)
    }


    logging.info("="*60)
    logging.info("MNIST Training Configuration")
    logging.info("="*60)
    for key, value in config.items():
        logging.info(f"{key}: {value}")
    logging.info("="*60)

    # Load data
    logging.info("\nLoading MNIST dataset...")
    train_loader, test_loader = load_mnist_data(config['batch_size'])
    logging.info(f"Train samples: {len(train_loader.dataset)}")
    logging.info(f"Test samples: {len(test_loader.dataset)}")


    # Print model summary
    logging.info("\n" + "="*60)
    logging.info("MODEL ARCHITECTURE SUMMARY")
    logging.info("="*60)
    logging.info(f"Model parameters: {count_parameters(model):,}")
    logging.info(f"Dropout rates: {config['dropout']}")
    logging.info("\nDetailed Model Summary:")
    try:
        summary(model, (1, 28, 28))
    except:
        logging.info("torchsummary not available, showing model structure:")
        logging.info(model)

    # Initialize optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config['factor'],
        patience=config['patience'],
        min_lr=config['min_lr']
    )

    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # scheduler = StepLR(optimizer, step_size=6, gamma=0.1)

    criterion = nn.CrossEntropyLoss()

    # Training history
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    learning_rates = []

    logging.info("\n" + "="*60)
    logging.info("TRAINING STARTED")
    logging.info("="*60)

    start_time = time.time()

    for epoch in range(config['epochs']):
        logging.info(f"\nEpoch [{epoch+1}/{config['epochs']}]")
        logging.info(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

        # Test
        test_loss, test_acc = test_epoch(model, test_loader, criterion, device)

        # Update scheduler
        scheduler.step(test_loss)

        # Store history
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
        learning_rates.append(optimizer.param_groups[0]['lr'])

        # Print epoch summary
        logging.info("\n" + "-"*50)
        logging.info("EPOCH SUMMARY")
        logging.info("-"*50)
        logging.info(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        logging.info(f"Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc:.2f}%")
        logging.info(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        logging.info("-"*50)

    # Final results
    total_time = time.time() - start_time
    logging.info("\n" + "="*60)
    logging.info("TRAINING COMPLETED")
    logging.info("="*60)
    logging.info(f"Total training time: {total_time:.2f} seconds")
    logging.info(f"Best train accuracy: {max(train_accuracies):.2f}%")
    logging.info(f"Best test accuracy: {max(test_accuracies):.2f}%")
    logging.info(f"Final train accuracy: {train_accuracies[-1]:.2f}%")
    logging.info(f"Final test accuracy: {test_accuracies[-1]:.2f}%")

    # Plot training history
    plt.figure(figsize=(15, 5))

    # Loss plot
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss', marker='o')
    plt.plot(test_losses, label='Test Loss', marker='s')
    plt.title('Training and Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Accuracy plot
    plt.subplot(1, 3, 2)
    plt.plot(train_accuracies, label='Train Accuracy', marker='o')
    plt.plot(test_accuracies, label='Test Accuracy', marker='s')
    plt.title('Training and Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    # Learning rate plot
    plt.subplot(1, 3, 3)
    plt.plot(learning_rates, label='Learning Rate', marker='o')
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')

    plt.tight_layout()
    # plt.show() # Commented out to prevent displaying plots

    # Save plots to the model-specific directory
    plt.figure(figsize=(15, 5)) # Re-create figure for saving
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss', marker='o')
    plt.plot(test_losses, label='Test Loss', marker='s')
    plt.title('Training and Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(model_output_dir, 'loss.png'))
    plt.close() # Close the plot to free memory

    plt.figure(figsize=(15, 5)) # Re-create figure for saving
    plt.subplot(1, 3, 2)
    plt.plot(train_accuracies, label='Train Accuracy', marker='o')
    plt.plot(test_accuracies, label='Test Accuracy', marker='s')
    plt.title('Training and Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(model_output_dir, 'accuracy.png'))
    plt.close() # Close the plot to free memory

    plt.figure(figsize=(15, 5)) # Re-create figure for saving
    plt.subplot(1, 3, 3)
    plt.plot(learning_rates, label='Learning Rate', marker='o')
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    plt.savefig(os.path.join(model_output_dir, 'learning_rate.png'))
    plt.close() # Close the plot to free memory

    # Save model checkpoint to the model-specific directory
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'train_accuracies': train_accuracies,
        'test_accuracies': test_accuracies,
        'train_losses': train_losses,
        'test_losses': test_losses
    }, os.path.join(model_output_dir, 'mnist_model_checkpoint.pth'))

    logging.info(f"\nModel and plots saved in '{model_output_dir}/'")

    return model, train_accuracies, test_accuracies

if __name__ == "__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Train MNIST model")
    parser.add_argument('--model', type=str, default='model1', choices=['model1', 'model2', 'model3'],
                        help='Model to use for training (default: model1)')

    args = parser.parse_args()

    # Clear any existing handlers to prevent duplicate logs if run multiple times
    logging.getLogger().handlers.clear()

    # Configure logging
    log_filename = f"{args.model}.log"
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler(log_filename),
                            logging.StreamHandler()
                        ])
    logging.info(f"Starting training for model: {args.model}")

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # Create a directory for saving model-specific outputs
    model_output_dir = args.model  # Directory name will be the model name
    os.makedirs(model_output_dir, exist_ok=True) # Create the directory if it doesn't exist
    logging.info(f"Created output directory: {model_output_dir}/")

    # Initialize model based on the argument
    if args.model == 'model1':
        model = MNISTModelLL().to(device)
    elif args.model == 'model2':
        model = MNISTModelGAP().to(device)
    elif args.model == 'model3':
        model = MNISTModelFinal().to(device)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    model, train_acc, test_acc = main(model)