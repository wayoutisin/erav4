# Import required libraries for deep learning and data processing
import torch  # PyTorch core library
import torch.nn as nn  # Neural network modules
import torch.nn.functional as F  # Functional operations
import torch.optim as optim  # Optimization algorithms
import torchvision  # Computer vision utilities
import torchvision.transforms as transforms  # Image transformations
from torch.utils.data import DataLoader  # Data loading utilities
import matplotlib.pyplot as plt  # Plotting library
import numpy as np  # Numerical computing
import time  # Time measurement utilities
import albumentations as A  # Advanced image augmentation library
from torchvision.datasets import CIFAR10  # CIFAR-10 dataset
from cifar10_album import CIFAR10Albumentations  # Custom dataset with albumentations
from model import FinalCNN  # Import our custom CNN model
import logging  # Logging utilities
import os  # Operating system interface


# Configure logging
def setup_logging(log_file='training.log'):
    """
    Set up logging configuration to write to both file and console.
    
    Args:
        log_file: Path to the log file
    """
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/{log_file}'),
            logging.StreamHandler()  # Also print to console
        ]
    )
    return logging.getLogger(__name__)

# Initialize logger
logger = setup_logging()

# Configure computing device (GPU if available, otherwise CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# CIFAR-10 class names in order (corresponding to labels 0-9)
cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']

# Data loading function with optional augmentation support
def load_cifar10_data(augment=''):
    """
    Load CIFAR-10 dataset with optional augmentation strategies.
    
    Args:
        augment: Augmentation strategy ('albumentations' or empty string for basic)
        
    Returns:
        tuple: (trainloader, testloader) PyTorch DataLoader objects
    """
    
    # Basic transformation pipeline for training data
    # Converts PIL images to tensors and normalizes using CIFAR-10 statistics
    transform_train = transforms.Compose([
        transforms.ToTensor(),  # Convert PIL Image to tensor (0-1 range)
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # Normalize to zero mean, unit variance
    ])

    # Basic transformation pipeline for test/validation data
    # Same normalization as training to ensure consistency
    transform_test = transforms.Compose([
        transforms.ToTensor(),  # Convert PIL Image to tensor
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # Same normalization
    ])
    
    # Choose dataset based on augmentation strategy
    if augment == 'albumentations':
        logger.info("Using albumentations for augmentation")
        # Use custom dataset with advanced augmentation capabilities
        train_dataset = CIFAR10Albumentations(
            root="./tdata/album", train=True, download=True, use_class_specific_aug=False
        )
        test_dataset = CIFAR10Albumentations(
            root="./tdata/album", train=False, download=True
        )
    else:
        # Use standard PyTorch CIFAR-10 dataset with basic transforms
        train_dataset = torchvision.datasets.CIFAR10(
            root="./tdata", train=True, download=True, transform=transform_train
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root="./tdata", train=False, download=True, transform=transform_test
        )

    # Create data loaders for efficient batch processing
    trainloader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    testloader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)
    
    return trainloader, testloader

# Receptive field calculation utility for CNN analysis
def calculate_receptive_field(model, input_size=(3, 32, 32)):
    """
    Calculate the approximate receptive field of a CNN model.
    
    The receptive field is the region in the input space that affects a particular
    output pixel. This is important for understanding what spatial context the model sees.
    
    Args:
        model: PyTorch CNN model
        input_size: Input tensor dimensions (channels, height, width)
        
    Returns:
        int: Approximate receptive field size in pixels
    """
    from collections import deque

    # Helper class to store layer information
    class LayerInfo:
        def __init__(self, kernel_size, stride, padding):
            self.kernel = kernel_size
            self.stride = stride
            self.padding = padding

    layers = []
    
    # Hook function to extract layer parameters
    def register_hook(module):
        if isinstance(module, (nn.Conv2d, nn.MaxPool2d)):
            k = module.kernel_size
            s = module.stride
            p = module.padding
            # Handle tuple parameters (convert to single values)
            if isinstance(k, tuple):
                k = k[0]
            if isinstance(s, tuple):
                s = s[0]
            if isinstance(p, tuple):
                p = p[0]
            layers.append(LayerInfo(k, s, p))

    # Apply hook to all modules in the model
    model.apply(register_hook)

    # Calculate receptive field using the formula:
    # RF_new = (RF_old - 1) * stride + kernel_size
    rf = 1  # Start with 1 pixel receptive field
    stride = 1  # Cumulative stride
    for layer in reversed(layers):  # Process layers in reverse order
        rf = ((rf - 1) * layer.stride) + layer.kernel
        stride *= layer.stride

    logger.info(f"Approximate Receptive Field: {rf}x{rf}")
    return rf


# Training function for one epoch
def train_epoch(model, trainloader, criterion, optimizer, device, scheduler=None):
    """
    Train the model for one complete epoch.
    
    Args:
        model: PyTorch model to train
        trainloader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimization algorithm
        device: Computing device (CPU/GPU)
        scheduler: Optional learning rate scheduler
        
    Returns:
        float: Training accuracy for this epoch
    """
    model.train()  # Set model to training mode (enables dropout, batch norm updates)
    running_loss = 0.0  # Accumulate loss for logging
    correct = 0  # Count correct predictions
    total = 0  # Count total samples processed
    
    # Iterate through all training batches
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # Move data to the specified device (GPU/CPU)
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass: compute predictions and loss
        optimizer.zero_grad()  # Clear gradients from previous iteration
        outputs = model(inputs)  # Forward pass through the model
        loss = criterion(outputs, targets)  # Compute loss
        
        # Backward pass: compute gradients and update weights
        loss.backward()  # Compute gradients
        optimizer.step()  # Update model parameters
        
        # Accumulate statistics for monitoring
        running_loss += loss.item()  # Add current batch loss
        _, predicted = outputs.max(1)  # Get predicted class (highest probability)
        total += targets.size(0)  # Count samples in this batch
        correct += predicted.eq(targets).sum().item()  # Count correct predictions
        
        # Log progress every 100 batches
        if batch_idx % 100 == 99:
            logger.info(f'Batch [{batch_idx+1}/{len(trainloader)}], '
                       f'Loss: {running_loss/100:.4f}, '
                       f'Acc: {100.*correct/total:.2f}%')
            running_loss = 0.0  # Reset running loss for next 100 batches
    
    # Calculate final training accuracy for this epoch
    train_acc = 100. * correct / total
    return train_acc

# Validation function for model evaluation
def validate(model, testloader, criterion, device):
    """
    Validate the model on test/validation data.
    
    Args:
        model: PyTorch model to evaluate
        testloader: DataLoader for test/validation data
        criterion: Loss function
        device: Computing device (CPU/GPU)
        
    Returns:
        tuple: (test_accuracy, average_loss)
    """
    model.eval()  # Set model to evaluation mode (disables dropout, batch norm uses running stats)
    test_loss = 0.0  # Accumulate loss
    correct = 0  # Count correct predictions
    total = 0  # Count total samples
    
    # Disable gradient computation for efficiency during validation
    with torch.no_grad():
        for inputs, targets in testloader:
            # Move data to the specified device
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass only (no backward pass needed)
            outputs = model(inputs)  # Get model predictions
            loss = criterion(outputs, targets)  # Compute loss
            
            # Accumulate statistics
            test_loss += loss.item()  # Add current batch loss
            _, predicted = outputs.max(1)  # Get predicted class
            total += targets.size(0)  # Count samples in batch
            correct += predicted.eq(targets).sum().item()  # Count correct predictions
    
    # Calculate final metrics
    test_acc = 100. * correct / total  # Validation accuracy percentage
    avg_loss = test_loss / len(testloader)  # Average loss per batch
    
    return test_acc, avg_loss

# Model parameter counting utility
def count_parameters(model):
    """
    Count the number of parameters in a PyTorch model.
    
    Args:
        model: PyTorch model
        
    Returns:
        tuple: (total_parameters, trainable_parameters)
    """
    total_params = sum(p.numel() for p in model.parameters())  # Count all parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)  # Count trainable parameters only
    return total_params, trainable_params

# Training progress visualization function
def plot_training_progress(train_accuracies, val_accuracies, train_losses, val_losses, save_dir='model'):
    """
    Create plots to visualize training progress over epochs and save to model directory.
    
    Args:
        train_accuracies: List of training accuracies per epoch
        val_accuracies: List of validation accuracies per epoch
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        save_dir: Directory to save the plots
    """
    # Create model directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))  # Create subplot figure
    
    epochs = range(1, len(train_accuracies) + 1)  # Epoch numbers for x-axis
    
    # Plot 1: Accuracy curves
    ax1.plot(epochs, train_accuracies, 'b-o', label='Training Accuracy')  # Blue line with circles
    ax1.plot(epochs, val_accuracies, 'r-o', label='Validation Accuracy')  # Red line with circles
    ax1.set_title('Training and Validation Accuracy')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy (%)')
    ax1.legend()  # Show legend
    ax1.grid(True)  # Add grid for easier reading
    
    # Plot 2: Loss curves
    ax2.plot(epochs, train_losses, 'b-o', label='Training Loss')  # Blue line with circles
    ax2.plot(epochs, val_losses, 'r-o', label='Validation Loss')  # Red line with circles
    ax2.set_title('Training and Validation Loss')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.legend()  # Show legend
    ax2.grid(True)  # Add grid for easier reading
    
    plt.tight_layout()  # Adjust layout to prevent overlap
    
    # Save plots to model directory
    accuracy_plot_path = os.path.join(save_dir, 'training_accuracy.png')
    loss_plot_path = os.path.join(save_dir, 'training_loss.png')
    combined_plot_path = os.path.join(save_dir, 'training_progress.png')
    
    plt.savefig(combined_plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Training progress plots saved to {combined_plot_path}")
    
    plt.close()  # Close the figure to free memory

# Main training loop
def main(model, epochs=10, augment='', decay=False, schedule=None, lr=0.001):
    logger.info("="*60)
    logger.info("BASELINE CNN FOR CIFAR-10")
    logger.info("="*60)
    
    # Load data
    logger.info("Loading CIFAR-10 dataset...")
    trainloader, testloader = load_cifar10_data(augment)
    
    logger.info(f"Training samples: {len(trainloader.dataset)}")
    logger.info(f"Test samples: {len(testloader.dataset)}")
    
    
    # Count parameters
    total_params, trainable_params = count_parameters(model)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Log model architecture
    logger.info("\nModel Architecture:")
    logger.info(str(model))
    
    # Compute and log receptive field
    _ = calculate_receptive_field(model)

    # Loss function and optimizer (simple Adam, no scheduler)
    criterion = nn.CrossEntropyLoss()
    logger.info(f"Using learning rate {lr}")
    if decay:
        logger.info(f"\nOptimizer: Adam (lr={lr})")
        optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=lr,
                    weight_decay=0.0001)
    else:
        # optimizer = optim.Adam(model.parameters(), lr=lr)
        logger.info(f"\nOptimizer: Cosine (lr={lr})")
        optimizer = torch.optim.SGD(
                        model.parameters(),
                        lr=lr,
                        momentum=0.9,
                        weight_decay=5e-4
                    )
    
    if schedule == 'step':
        logger.info(f"Using step scheduler")
        scheduler = optim.lr_scheduler.StepLR(
                            optimizer,
                            step_size=7,
                            gamma=0.5
                        )
    elif schedule == 'cosine':
        logger.info(f"Using cosine scheduler")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=1e-4
        )
    
    
    logger.info(f"Loss function: CrossEntropyLoss")
    logger.info(f"Training epochs: {epochs}")
    
    # Training history
    train_accuracies = []
    val_accuracies = []
    train_losses = []
    val_losses = []
    
    logger.info("\n" + "="*60)
    logger.info("STARTING TRAINING")
    logger.info("="*60)
    
    # Training loop for specified epochs
    for epoch in range(epochs):
        start_time = time.time()
        
        logger.info(f"\nEpoch [{epoch+1}/{epochs}]")
        logger.info("-" * 40)
        
        # Train for one epoch
        train_acc = train_epoch(model, trainloader, criterion, optimizer, device)
        
        # Validate
        val_acc, val_loss = validate(model, testloader, criterion, device)
        
        # Calculate approximate training loss for plotting
        model.eval()
        train_loss = 0.0
        with torch.no_grad():
            for inputs, targets in trainloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                train_loss += criterion(outputs, targets).item()
        train_loss /= len(trainloader)
        
        # Store metrics
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Log epoch results
        epoch_time = time.time() - start_time
        logger.info(f"\nEpoch {epoch+1} Summary:")
        logger.info(f"Train Accuracy: {train_acc:.2f}%")
        logger.info(f"Validation Accuracy: {val_acc:.2f}%")
        logger.info(f"Train Loss: {train_loss:.4f}")
        logger.info(f"Validation Loss: {val_loss:.4f}")
        logger.info(f"Time: {epoch_time:.1f}s")

        # scheduler step
        if scheduler:
            scheduler.step()
    
    # Final results
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETED")
    logger.info("="*60)
    
    best_val_acc = max(val_accuracies)
    best_epoch = val_accuracies.index(best_val_acc) + 1
    final_val_acc = val_accuracies[-1]
    
    logger.info(f"Final Validation Accuracy: {final_val_acc:.2f}%")
    logger.info(f"Best Validation Accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})")
    logger.info(f"Total Parameters: {total_params:,}")
    
    # Generate and save training plots
    logger.info("\nGenerating training plots...")
    plot_training_progress(train_accuracies, val_accuracies, train_losses, val_losses)
    
    # Test on individual classes
    logger.info("\nPer-class accuracy:")
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    
    model.eval()
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == targets).squeeze()
            for i in range(targets.size(0)):
                label = targets[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    for i in range(10):
        if class_total[i] > 0:
            accuracy = 100 * class_correct[i] / class_total[i]
            logger.info(f'{cifar10_classes[i]}: {accuracy:.1f}%')
    
    logger.info(f"\nBaseline model performance summary:")
    logger.info(f"• Architecture: 5-block CNN (ResNet-inspired)")
    logger.info(f"• No data augmentation")
    logger.info(f"• No max pooling (used stride and global avg pooling)")
    logger.info(f"• Simple SGD optimizer (cosine scheduler)")
    logger.info(f"• Final accuracy: {final_val_acc:.2f}%")
    logger.info(f"• Parameters: {total_params:,}")
    
    return model, train_accuracies, val_accuracies



if __name__ == "__main__":
    # Initialize model
    logger.info("\nInitializing Final CNN Model...")
    model = FinalCNN(num_classes=10).to(device)

    trained_model, train_accs, val_accs = main(model=model, epochs=30, augment='albumentations', schedule='cosine', lr=0.05)
