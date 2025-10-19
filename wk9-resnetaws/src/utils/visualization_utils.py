"""
Visualization utilities for ResNet18 training
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from typing import List, Dict, Any, Optional
from pathlib import Path

from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd


def plot_training_history(history: Dict[str, List[float]], 
                         save_path: Optional[str] = None,
                         show_plot: bool = True) -> None:
    """Plot training history curves"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss plot
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Training Loss', marker='o')
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss', marker='s')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epochs')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', marker='o')
    axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', marker='s')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].set_xlabel('Epochs')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Learning rate plot (if available)
    if 'lr' in history:
        axes[1, 0].plot(epochs, history['lr'], 'g-', marker='o')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_xlabel('Epochs')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'Learning Rate\nData Not Available', 
                       ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Learning Rate Schedule')
    
    # Loss difference plot
    loss_diff = np.array(history['val_loss']) - np.array(history['train_loss'])
    axes[1, 1].plot(epochs, loss_diff, 'purple', marker='o')
    axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[1, 1].set_title('Overfitting Indicator (Val Loss - Train Loss)')
    axes[1, 1].set_xlabel('Epochs')
    axes[1, 1].set_ylabel('Loss Difference')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_confusion_matrix(y_true: List[int], 
                         y_pred: List[int], 
                         class_names: List[str],
                         save_path: Optional[str] = None,
                         show_plot: bool = True) -> None:
    """Plot confusion matrix"""
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_class_accuracy(y_true: List[int], 
                       y_pred: List[int], 
                       class_names: List[str],
                       save_path: Optional[str] = None,
                       show_plot: bool = True) -> None:
    """Plot per-class accuracy"""
    
    cm = confusion_matrix(y_true, y_pred)
    class_accuracy = cm.diagonal() / cm.sum(axis=1)
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(class_names, class_accuracy, color='skyblue', alpha=0.7)
    plt.title('Per-Class Accuracy')
    plt.xlabel('Classes')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, class_accuracy):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def visualize_predictions(images: torch.Tensor,
                         true_labels: torch.Tensor,
                         predicted_labels: torch.Tensor,
                         class_names: List[str],
                         confidences: Optional[torch.Tensor] = None,
                         num_images: int = 8,
                         save_path: Optional[str] = None,
                         show_plot: bool = True) -> None:
    """Visualize predictions on sample images"""
    
    # Denormalize images for display
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    images = images * std + mean
    images = torch.clamp(images, 0, 1)
    
    num_images = min(num_images, len(images))
    cols = 4
    rows = (num_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for i in range(num_images):
        img = images[i].permute(1, 2, 0).cpu().numpy()
        true_class = class_names[true_labels[i]]
        pred_class = class_names[predicted_labels[i]]
        
        # Determine color based on correctness
        color = 'green' if true_labels[i] == predicted_labels[i] else 'red'
        
        axes[i].imshow(img)
        axes[i].axis('off')
        
        title = f'True: {true_class}\nPred: {pred_class}'
        if confidences is not None:
            title += f'\nConf: {confidences[i]:.3f}'
        
        axes[i].set_title(title, color=color, fontsize=10)
    
    # Hide unused subplots
    for i in range(num_images, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_model_architecture_summary(model_summary: Dict[str, Any],
                                  save_path: Optional[str] = None,
                                  show_plot: bool = True) -> None:
    """Plot model architecture summary"""
    
    layers_df = pd.DataFrame(model_summary['layers'])
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Parameters distribution by layer type
    layer_types = layers_df['type'].value_counts()
    axes[0, 0].pie(layer_types.values, labels=layer_types.index, autopct='%1.1f%%')
    axes[0, 0].set_title('Layer Types Distribution')
    
    # Parameters by layer
    top_layers = layers_df.nlargest(10, 'parameters')
    axes[0, 1].barh(range(len(top_layers)), top_layers['parameters'])
    axes[0, 1].set_yticks(range(len(top_layers)))
    axes[0, 1].set_yticklabels(top_layers['name'], fontsize=8)
    axes[0, 1].set_xlabel('Parameters')
    axes[0, 1].set_title('Top 10 Layers by Parameters')
    
    # Trainable vs Frozen parameters
    trainable_params = model_summary['trainable_parameters']
    frozen_params = model_summary['frozen_parameters']
    
    axes[1, 0].pie([trainable_params, frozen_params], 
                   labels=['Trainable', 'Frozen'],
                   autopct='%1.1f%%',
                   colors=['lightgreen', 'lightcoral'])
    axes[1, 0].set_title('Trainable vs Frozen Parameters')
    
    # Model statistics
    stats_text = f"""Model: {model_summary['model_name']}
Total Parameters: {model_summary['total_parameters']:,}
Trainable Parameters: {model_summary['trainable_parameters']:,}
Frozen Parameters: {model_summary['frozen_parameters']:,}
"""
    axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes,
                    fontsize=12, verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    axes[1, 1].axis('off')
    axes[1, 1].set_title('Model Statistics')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_data_distribution(dataset_info: Dict[str, Any],
                          save_path: Optional[str] = None,
                          show_plot: bool = True) -> None:
    """Plot dataset statistics and distribution"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Dataset split distribution
    split_sizes = [dataset_info['train_size'], dataset_info['val_size']]
    split_labels = ['Training', 'Validation']
    
    axes[0, 0].pie(split_sizes, labels=split_labels, autopct='%1.1f%%',
                   colors=['lightblue', 'lightcoral'])
    axes[0, 0].set_title('Dataset Split Distribution')
    
    # Class distribution (if available)
    if 'class_distribution' in dataset_info:
        class_dist = dataset_info['class_distribution']
        axes[0, 1].bar(class_dist.keys(), class_dist.values(), alpha=0.7)
        axes[0, 1].set_title('Class Distribution')
        axes[0, 1].set_xlabel('Classes')
        axes[0, 1].set_ylabel('Number of Samples')
        axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Dataset statistics
    stats_text = f"""Dataset Statistics:
Total Samples: {dataset_info.get('total_samples', 'N/A'):,}
Training Samples: {dataset_info.get('train_size', 'N/A'):,}
Validation Samples: {dataset_info.get('val_size', 'N/A'):,}
Number of Classes: {dataset_info.get('num_classes', 'N/A')}
Image Size: {dataset_info.get('image_size', 'N/A')}
"""
    
    axes[1, 0].text(0.1, 0.5, stats_text, transform=axes[1, 0].transAxes,
                    fontsize=12, verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    axes[1, 0].axis('off')
    axes[1, 0].set_title('Dataset Information')
    
    # Hide unused subplot
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def create_training_report(history: Dict[str, List[float]],
                          y_true: List[int],
                          y_pred: List[int],
                          class_names: List[str],
                          model_summary: Dict[str, Any],
                          output_dir: str) -> None:
    """Create comprehensive training report with all visualizations"""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Training history plot
    plot_training_history(
        history, 
        save_path=str(output_path / 'training_history.png'),
        show_plot=False
    )
    
    # Confusion matrix
    plot_confusion_matrix(
        y_true, y_pred, class_names,
        save_path=str(output_path / 'confusion_matrix.png'),
        show_plot=False
    )
    
    # Per-class accuracy
    plot_class_accuracy(
        y_true, y_pred, class_names,
        save_path=str(output_path / 'class_accuracy.png'),
        show_plot=False
    )
    
    # Model architecture summary
    plot_model_architecture_summary(
        model_summary,
        save_path=str(output_path / 'model_summary.png'),
        show_plot=False
    )
    
    print(f"Training report saved to: {output_path}")


def denormalize_image(tensor: torch.Tensor, 
                     mean: List[float] = [0.485, 0.456, 0.406],
                     std: List[float] = [0.229, 0.224, 0.225]) -> torch.Tensor:
    """Denormalize image tensor for visualization"""
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return torch.clamp(tensor * std + mean, 0, 1)