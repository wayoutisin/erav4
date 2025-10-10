import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import ResNet56
import logging

# Configure logging
logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check device (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f'Using device: {device}')
if torch.cuda.is_available():
    logger.info(f'GPU: {torch.cuda.get_device_name(0)}')

# CIFAR-100 mean and std for normalization
mean = [0.5071, 0.4867, 0.4408]
std = [0.2675, 0.2565, 0.2761]

class Cutout:
    """
    Custom Cutout augmentation: randomly mask out square regions of input image.
    """
    def __init__(self, n_holes=1, length=16):
        self.n_holes = n_holes
        self.length = length
    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            mask[y1:y2, x1:x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask
        return img

# Compose data augmentations for training and testing
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4), # Randomly crop 32x32 with padding
    transforms.RandomHorizontalFlip(),    # Randomly flip horizontally
    transforms.RandomRotation(15),        # Slightly random rotate
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Color/tone augment
    transforms.ToTensor(),
    transforms.Normalize(mean, std),      # Normalize to CIFAR-100
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
    Cutout(n_holes=1, length=16),         # Custom cutout augmentation
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

# Download and prepare the datasets
train_dataset = torchvision.datasets.CIFAR100(
    root='./data', train=True, download=True, transform=train_transform
)
test_dataset = torchvision.datasets.CIFAR100(
    root='./data', train=False, download=True, transform=test_transform
)

# Create data loaders for batch processing
batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

# Training hyperparameters
target_acc = 73.0
num_classes = 100
num_epochs = 150
initial_lr = 0.1
weight_decay = 5e-4
momentum = 0.9

def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Trains the model for a single epoch over the train_loader.
    Returns average loss and accuracy for the epoch.
    """
    model.train()
    running_loss = 0.0
    correct, total = 0, 0
    pbar = tqdm(train_loader, desc='Training')
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()          # Zero gradients
        outputs = model(inputs)        # Forward pass
        loss = criterion(outputs, targets) # Compute loss
        loss.backward()                # Backward pass
        optimizer.step()               # Update weights
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        pbar.set_postfix({'loss': f'{running_loss/len(pbar):.3f}', 'acc': f'{100.*correct/total:.2f}%'})
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def validate(model, test_loader, criterion, device):
    """
    Evaluates the model on test/validation set. Returns average loss and accuracy.
    """
    model.eval()
    running_loss = 0.0
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc='Validation'):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    epoch_loss = running_loss / len(test_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def save_checkpoint(model, optimizer, epoch, acc, filename='model/best_model.pth'):
    """
    Saves model and optimizer state at current epoch.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': acc,
    }
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(checkpoint, filename)
    logger.info(f'Checkpoint saved: {filename}')

def main():
    # Create ResNet-56 model for CIFAR-100
    model = ResNet56(num_classes=num_classes).to(device)
    # Cross-entropy loss and SGD optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=initial_lr,
                         momentum=momentum, weight_decay=weight_decay)
    # Cosine annealing LR scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)
    # Store training/validation curves (optional for later plotting)
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}
    best_acc = 0.0
    logger.info('Starting training...')
    logger.info('='*80)
    # Training loop
    for epoch in range(num_epochs):
        logger.info(f'\nEpoch [{epoch+1}/{num_epochs}]')
        logger.info(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        # Train for one epoch
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        # Evaluate on test data
        val_loss, val_acc = validate(model, test_loader, criterion, device)
        scheduler.step()  # Adjust learning rate
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        logger.info(f'\nEpoch {epoch+1} Summary:')
        logger.info(f'  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        logger.info(f'  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%')
        # Save the best model so far
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(model, optimizer, epoch+1, val_acc, 'model/best_model.pth')
            logger.info(f'  ✓ New best accuracy: {best_acc:.2f}%')
        # Stop early if target accuracy reached
        if val_acc >= target_acc:
            logger.info(f'\n🎉 Target accuracy of {target_acc}% reached!')
            logger.info(f'Best validation accuracy: {best_acc:.2f}%')
            break
        logger.info('-'*80)
    logger.info('\n' + '='*80)
    logger.info('Training completed!')
    logger.info(f'Best validation accuracy: {best_acc:.2f}%')

if __name__ == "__main__":
    main()
