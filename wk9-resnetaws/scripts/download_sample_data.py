#!/usr/bin/env python3
"""
Download sample dataset for ResNet18 training.
Creates a small CIFAR-10 style dataset for testing.
"""

import os
import zipfile
from pathlib import Path
import shutil
from PIL import Image
import numpy as np

def create_synthetic_dataset(output_dir, num_classes=10, samples_per_class=100):
    """Create a synthetic dataset with colored patterns"""
    
    output_dir = Path(output_dir)
    train_dir = output_dir / 'train'
    val_dir = output_dir / 'val'
    
    # Create directories
    for class_idx in range(num_classes):
        class_name = f'class{class_idx}'
        (train_dir / class_name).mkdir(parents=True, exist_ok=True)
        (val_dir / class_name).mkdir(parents=True, exist_ok=True)
    
    print(f"Creating synthetic dataset with {num_classes} classes...")
    
    # Generate images for each class
    for class_idx in range(num_classes):
        class_name = f'class{class_idx}'
        
        # Generate unique color pattern for each class
        hue = (class_idx * 36) % 360  # Spread hues across spectrum
        
        # Training images
        train_samples = int(samples_per_class * 0.8)
        for img_idx in range(train_samples):
            img = generate_pattern_image(hue, 224, 224, variation=img_idx)
            img_path = train_dir / class_name / f'{img_idx:04d}.png'
            img.save(img_path)
        
        # Validation images  
        val_samples = samples_per_class - train_samples
        for img_idx in range(val_samples):
            img = generate_pattern_image(hue, 224, 224, variation=img_idx + 1000)
            img_path = val_dir / class_name / f'{img_idx:04d}.png'
            img.save(img_path)
        
        print(f"  Generated {train_samples} train + {val_samples} val images for {class_name}")
    
    print(f"✓ Synthetic dataset created at: {output_dir}")
    return True

def generate_pattern_image(hue, width, height, variation=0):
    """Generate a pattern image with specific hue"""
    
    # Create base pattern
    np.random.seed(variation)
    
    # Generate geometric pattern
    x = np.linspace(0, 4*np.pi, width)
    y = np.linspace(0, 4*np.pi, height)
    X, Y = np.meshgrid(x, y)
    
    # Create pattern based on hue
    pattern1 = np.sin(X + variation * 0.1) * np.cos(Y + variation * 0.1)
    pattern2 = np.sin(X * 2 + variation * 0.2) + np.cos(Y * 2 + variation * 0.2)
    
    # Combine patterns
    combined = (pattern1 + pattern2) * 0.5
    
    # Normalize to 0-1
    combined = (combined - combined.min()) / (combined.max() - combined.min())
    
    # Convert to RGB using HSV
    from colorsys import hsv_to_rgb
    
    # Create RGB image
    rgb_array = np.zeros((height, width, 3), dtype=np.uint8)
    
    for i in range(height):
        for j in range(width):
            # Use pattern to vary saturation and value
            saturation = 0.7 + 0.3 * combined[i, j]
            value = 0.5 + 0.5 * combined[i, j]
            
            # Convert HSV to RGB
            r, g, b = hsv_to_rgb(hue/360.0, saturation, value)
            rgb_array[i, j] = [int(r*255), int(g*255), int(b*255)]
    
    # Add some noise for variation
    noise = np.random.randint(-20, 21, rgb_array.shape)
    rgb_array = np.clip(rgb_array.astype(int) + noise, 0, 255).astype(np.uint8)
    
    return Image.fromarray(rgb_array)

def download_cifar10_sample():
    """Download a sample of CIFAR-10 dataset"""
    print("CIFAR-10 download requires torchvision.")
    print("Run: pip install torchvision")
    print("Then use the synthetic dataset for now.")
    return False

def create_readme(output_dir):
    """Create README for the dataset"""
    readme_content = """# Sample Dataset for ResNet18 Training

This is a synthetic dataset created for testing the ResNet18 training pipeline.

## Structure

```
sample_dataset/
├── train/
│   ├── class0/ (80 images)
│   ├── class1/ (80 images)
│   ├── ...
│   └── class9/ (80 images)
└── val/
    ├── class0/ (20 images)  
    ├── class1/ (20 images)
    ├── ...
    └── class9/ (20 images)
```

## Dataset Details

- **Classes**: 10 synthetic classes (class0-class9)
- **Images per class**: 100 total (80 train, 20 validation)
- **Image size**: 224x224 pixels
- **Format**: PNG
- **Total size**: ~1000 images

## Class Descriptions

Each class has a unique color pattern:
- class0: Red-based patterns
- class1: Orange-based patterns  
- class2: Yellow-based patterns
- class3: Green-based patterns
- class4: Cyan-based patterns
- class5: Blue-based patterns
- class6: Purple-based patterns
- class7: Pink-based patterns
- class8: Brown-based patterns
- class9: Gray-based patterns

## Usage

This dataset is designed for:
1. Testing the training pipeline
2. Verifying model architecture
3. Quick sanity checks
4. Development and debugging

For real-world applications, replace with your actual dataset.

## Training Expectations

Since this is a synthetic dataset with clear visual patterns:
- Training should converge quickly (10-20 epochs)
- Expected accuracy: 80-95%
- Training from scratch should work well
- Good for testing overfitting and generalization
"""
    
    with open(Path(output_dir) / 'README.md', 'w') as f:
        f.write(readme_content)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Download or create sample dataset')
    parser.add_argument('--output-dir', type=str, default='data/sample_dataset', 
                       help='Output directory for dataset')
    parser.add_argument('--num-classes', type=int, default=10,
                       help='Number of classes to create')
    parser.add_argument('--samples-per-class', type=int, default=100,
                       help='Number of samples per class')
    parser.add_argument('--type', choices=['synthetic', 'cifar10'], default='synthetic',
                       help='Type of dataset to create')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    print("Sample Dataset Creator")
    print("=" * 30)
    print(f"Output directory: {output_dir}")
    print(f"Dataset type: {args.type}")
    
    # Check if directory exists
    if output_dir.exists():
        response = input(f"Directory {output_dir} exists. Overwrite? (y/N): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
        shutil.rmtree(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dataset
    if args.type == 'synthetic':
        success = create_synthetic_dataset(
            output_dir, 
            args.num_classes, 
            args.samples_per_class
        )
    else:
        success = download_cifar10_sample()
    
    if success:
        create_readme(output_dir)
        print(f"\n✅ Dataset created successfully!")
        print(f"📁 Location: {output_dir.absolute()}")
        print(f"\nNext steps:")
        print(f"1. Check dataset: python scripts/check_dataset.py --data-path {output_dir}")
        print(f"2. Run local test: python src/training/train.py --local-test")
        print(f"3. Full training: python src/training/train.py --from-scratch")
    else:
        print(f"\n❌ Dataset creation failed!")

if __name__ == "__main__":
    main()