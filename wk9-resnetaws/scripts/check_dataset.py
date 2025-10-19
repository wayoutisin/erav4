#!/usr/bin/env python3
"""
Dataset checker utility for ResNet18 training project.
Validates dataset structure and provides statistics.
"""

import os
import argparse
from pathlib import Path
from collections import defaultdict
import json

def check_dataset_structure(data_path):
    """Check if dataset has proper structure"""
    data_path = Path(data_path)
    
    if not data_path.exists():
        print(f"✗ Dataset path does not exist: {data_path}")
        return False
    
    # Look for train/val split or single directory
    train_dir = data_path / 'train'
    val_dir = data_path / 'val'
    
    has_split = train_dir.exists() and val_dir.exists()
    
    if has_split:
        print(f"✓ Found train/val split structure")
        return check_split_dataset(train_dir, val_dir)
    else:
        print(f"✓ Found single directory structure")
        return check_single_dataset(data_path)

def check_split_dataset(train_dir, val_dir):
    """Check train/val split dataset"""
    
    # Get class directories
    train_classes = [d.name for d in train_dir.iterdir() if d.is_dir()]
    val_classes = [d.name for d in val_dir.iterdir() if d.is_dir()]
    
    if not train_classes:
        print(f"✗ No class directories found in {train_dir}")
        return False
    
    if not val_classes:
        print(f"✗ No class directories found in {val_dir}")
        return False
    
    # Check class consistency
    train_set = set(train_classes)
    val_set = set(val_classes)
    
    if train_set != val_set:
        print(f"⚠ Warning: Train and validation classes don't match")
        print(f"  Train only: {train_set - val_set}")
        print(f"  Val only: {val_set - train_set}")
    
    all_classes = sorted(train_set | val_set)
    print(f"✓ Found {len(all_classes)} classes: {all_classes}")
    
    # Count images per class
    train_counts = count_images_per_class(train_dir, all_classes)
    val_counts = count_images_per_class(val_dir, all_classes)
    
    total_train = sum(train_counts.values())
    total_val = sum(val_counts.values())
    
    print(f"\nDataset Statistics:")
    print(f"  Training samples: {total_train}")
    print(f"  Validation samples: {total_val}")
    print(f"  Total samples: {total_train + total_val}")
    
    # Per-class breakdown
    print(f"\nPer-class breakdown:")
    for class_name in all_classes:
        train_count = train_counts.get(class_name, 0)
        val_count = val_counts.get(class_name, 0)
        total_count = train_count + val_count
        print(f"  {class_name}: {train_count} train, {val_count} val, {total_count} total")
    
    # Check for class imbalance
    class_totals = [train_counts.get(c, 0) + val_counts.get(c, 0) for c in all_classes]
    min_samples = min(class_totals)
    max_samples = max(class_totals)
    
    if max_samples > 2 * min_samples:
        print(f"⚠ Warning: Class imbalance detected")
        print(f"  Min samples per class: {min_samples}")
        print(f"  Max samples per class: {max_samples}")
        print(f"  Consider data augmentation or resampling")
    
    return total_train > 0 and total_val > 0

def check_single_dataset(data_path):
    """Check single directory dataset"""
    
    # Get class directories
    class_dirs = [d for d in data_path.iterdir() if d.is_dir()]
    
    if not class_dirs:
        print(f"✗ No class directories found in {data_path}")
        return False
    
    class_names = [d.name for d in class_dirs]
    print(f"✓ Found {len(class_names)} classes: {class_names}")
    
    # Count images per class
    class_counts = count_images_per_class(data_path, class_names)
    total_samples = sum(class_counts.values())
    
    print(f"\nDataset Statistics:")
    print(f"  Total samples: {total_samples}")
    print(f"  Will be split 80/20 for train/validation")
    
    # Per-class breakdown
    print(f"\nPer-class breakdown:")
    for class_name in class_names:
        count = class_counts.get(class_name, 0)
        train_est = int(count * 0.8)
        val_est = count - train_est
        print(f"  {class_name}: {count} total (~{train_est} train, ~{val_est} val)")
    
    return total_samples > 0

def count_images_per_class(base_dir, class_names):
    """Count images in each class directory"""
    counts = {}
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    for class_name in class_names:
        class_dir = base_dir / class_name
        if class_dir.exists():
            image_files = [f for f in class_dir.iterdir() 
                          if f.is_file() and f.suffix.lower() in image_extensions]
            counts[class_name] = len(image_files)
        else:
            counts[class_name] = 0
    
    return counts

def suggest_config_updates(data_path):
    """Suggest configuration updates based on dataset"""
    data_path = Path(data_path)
    
    # Count total classes
    train_dir = data_path / 'train'
    if train_dir.exists():
        classes = [d.name for d in train_dir.iterdir() if d.is_dir()]
    else:
        classes = [d.name for d in data_path.iterdir() if d.is_dir()]
    
    num_classes = len(classes)
    
    # Count total samples
    if train_dir.exists():
        train_counts = count_images_per_class(train_dir, classes)
        val_dir = data_path / 'val'
        val_counts = count_images_per_class(val_dir, classes) if val_dir.exists() else {}
        total_samples = sum(train_counts.values()) + sum(val_counts.values())
    else:
        class_counts = count_images_per_class(data_path, classes)
        total_samples = sum(class_counts.values())
    
    print(f"\nConfiguration Suggestions:")
    print(f"  Update 'num_classes' to: {num_classes}")
    print(f"  Update 'dataset_path' to: {data_path}")
    
    # Batch size suggestions
    if total_samples < 1000:
        print(f"  Suggested batch_size: 8-16 (small dataset)")
    elif total_samples < 10000:
        print(f"  Suggested batch_size: 16-32 (medium dataset)")
    else:
        print(f"  Suggested batch_size: 32-64 (large dataset)")
    
    # Training suggestions
    if total_samples < 500:
        print(f"  Suggested epochs: 100-200 (very small dataset)")
        print(f"  Consider data augmentation")
    elif total_samples < 5000:
        print(f"  Suggested epochs: 50-100 (small dataset)")
    else:
        print(f"  Suggested epochs: 20-50 (sufficient data)")

def create_sample_config(data_path, output_path):
    """Create a sample configuration file based on dataset"""
    data_path = Path(data_path)
    
    # Determine number of classes
    train_dir = data_path / 'train'
    if train_dir.exists():
        classes = [d.name for d in train_dir.iterdir() if d.is_dir()]
    else:
        classes = [d.name for d in data_path.iterdir() if d.is_dir()]
    
    num_classes = len(classes)
    
    # Create configuration
    config = {
        "data": {
            "dataset_path": str(data_path),
            "batch_size": 32,
            "num_workers": 2,
            "image_size": 224,
            "train_split": 0.8,
            "augmentation": {
                "random_resized_crop": True,
                "random_horizontal_flip": True,
                "color_jitter": {
                    "brightness": 0.1,
                    "contrast": 0.1,
                    "saturation": 0.1,
                    "hue": 0.1
                },
                "random_rotation": 10
            }
        },
        "model": {
            "name": "resnet18",
            "num_classes": num_classes,
            "pretrained": False,
            "custom_head": False,
            "dropout": 0.0,
            "auxiliary_classifier": False,
            "print_architecture": True
        },
        "training": {
            "epochs": 50,
            "optimizer": {
                "name": "adam",
                "lr": 0.001,
                "weight_decay": 0.0001
            },
            "scheduler": {
                "name": "step",
                "step_size": 20,
                "gamma": 0.1
            },
            "early_stopping": {
                "patience": 15,
                "min_delta": 0.001
            }
        },
        "output": {
            "output_dir": "outputs",
            "save_frequency": 5,
            "log_frequency": 10
        },
        "hardware": {
            "device": "auto",
            "mixed_precision": False,
            "compile_model": False
        }
    }
    
    # Save configuration
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✓ Sample configuration saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Check dataset structure and statistics')
    parser.add_argument('--data-path', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--create-config', type=str, help='Create sample config file at path')
    
    args = parser.parse_args()
    
    print("Dataset Structure Checker")
    print("=" * 50)
    
    # Check dataset structure
    is_valid = check_dataset_structure(args.data_path)
    
    if not is_valid:
        print("\n❌ Dataset validation failed!")
        print("\nExpected structure:")
        print("Option 1 (train/val split):")
        print("  data/")
        print("  ├── train/")
        print("  │   ├── class1/")
        print("  │   │   ├── image1.jpg")
        print("  │   │   └── image2.jpg")
        print("  │   └── class2/")
        print("  │       └── image1.jpg")
        print("  └── val/")
        print("      ├── class1/")
        print("      │   └── image1.jpg")
        print("      └── class2/")
        print("          └── image1.jpg")
        print("\nOption 2 (single directory):")
        print("  data/")
        print("  ├── class1/")
        print("  │   ├── image1.jpg")
        print("  │   └── image2.jpg")
        print("  └── class2/")
        print("      └── image1.jpg")
        return
    
    print("\n✅ Dataset validation passed!")
    
    # Provide configuration suggestions
    suggest_config_updates(args.data_path)
    
    # Create sample config if requested
    if args.create_config:
        create_sample_config(args.data_path, args.create_config)

if __name__ == "__main__":
    main()