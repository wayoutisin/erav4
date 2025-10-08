# Import required libraries for data augmentation and dataset handling
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.datasets import CIFAR10
import warnings
warnings.filterwarnings('ignore')

# CIFAR-10 normalization parameters (computed from training data)
# These values ensure pixel values are centered around 0 with unit variance
mean = (0.4914, 0.4822, 0.4465)  # RGB channel means
std = (0.2023, 0.1994, 0.2010)   # RGB channel standard deviations

# Standard augmentation pipeline for most CIFAR-10 classes
# This provides moderate augmentation to improve generalization without being too aggressive
standard_transform = A.Compose([
    A.HorizontalFlip(p=0.5),  # Randomly flip horizontally 50% of the time
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),  # Random geometric transformations
    A.CoarseDropout(max_holes=1, max_height=16, max_width=16,  # Random rectangular cutouts
                    min_holes=1, min_height=16, min_width=16,
                    fill_value=mean, mask_fill_value=None, p=0.5),  # Fill with dataset mean
    A.Normalize(mean=mean, std=std),  # Normalize to zero mean, unit variance
    ToTensorV2()  # Convert to PyTorch tensor
])

# Enhanced augmentation pipeline specifically for cats (class 3) and dogs (class 5)
# These classes typically have lower accuracy, so we apply stronger augmentation to improve learning
strong_transform = A.Compose([
    A.HorizontalFlip(p=0.5),  # Random horizontal flip
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.15, rotate_limit=20, p=0.6),  # More aggressive geometric transforms
    A.OneOf([  # Randomly choose one color transformation
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0),  # Adjust color properties
        A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1.0),  # Shift RGB channels independently
    ], p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),  # Additional brightness/contrast changes
    A.CoarseDropout(max_holes=1, max_height=16, max_width=16,  # Random rectangular cutouts
                    min_holes=1, min_height=16, min_width=16,
                    fill_value=mean, p=0.5),  # Fill with dataset mean
    A.OneOf([  # Randomly choose one blur effect
        A.MotionBlur(blur_limit=3, p=1.0),  # Simulate camera motion
        A.MedianBlur(blur_limit=3, p=1.0),  # Median filtering
        A.GaussianBlur(blur_limit=3, p=1.0),  # Gaussian smoothing
    ], p=0.3),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),  # Add Gaussian noise to simulate sensor noise
    A.Normalize(mean=mean, std=std),  # Normalize to zero mean, unit variance
    ToTensorV2()  # Convert to PyTorch tensor
])

# Test/validation transform - minimal processing for evaluation
# Only normalization and tensor conversion, no augmentation to ensure consistent evaluation
test_transform = A.Compose([
    A.Normalize(mean=mean, std=std),  # Normalize using same parameters as training
    ToTensorV2()  # Convert to PyTorch tensor
])


# Custom CIFAR-10 dataset class that uses Albumentations for data augmentation
class CIFAR10Albumentations(CIFAR10):
    """
    Extended CIFAR-10 dataset with class-specific augmentation strategies.
    
    This class inherits from torchvision's CIFAR10 and adds intelligent augmentation
    that applies different augmentation strengths based on the class label.
    """
    
    def __init__(self, root, train=True, download=False, 
                 use_class_specific_aug=True):
        """
        Initialize the dataset with optional class-specific augmentation.
        
        Args:
            root: Directory where dataset will be stored
            train: Whether to use training set (True) or test set (False)
            download: Whether to download dataset if not present
            use_class_specific_aug: Whether to apply stronger augmentation to cats/dogs
        """
        super().__init__(root=root, train=train, download=download)
        self.use_class_specific_aug = use_class_specific_aug and train
        
    def __getitem__(self, index):
        """
        Get a single sample from the dataset with appropriate augmentation.
        
        Args:
            index: Index of the sample to retrieve
            
        Returns:
            tuple: (augmented_image_tensor, label)
        """
        # Get original image and label
        image, label = self.data[index], self.targets[index]
        
        # Choose augmentation strategy based on class and training mode
        if self.use_class_specific_aug and label in [3, 5]:  # cats (3) and dogs (5)
            transform = strong_transform  # Use enhanced augmentation for challenging classes
        elif self.train:
            transform = standard_transform  # Use standard augmentation for other classes
        else:
            transform = test_transform  # No augmentation for validation/test
        
        # Apply the selected transformation pipeline
        augmented = transform(image=image)
        image = augmented["image"]
        return image, label