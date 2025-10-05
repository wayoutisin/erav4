import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision
from torchvision.datasets import CIFAR10
import warnings
warnings.filterwarnings('ignore')


mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)

albumentations_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
    A.CoarseDropout(max_holes=1, max_height=16, max_width=16,
                    min_holes=1, min_height=16, min_width=16,
                    fill_value=mean, mask_fill_value=None, p=0.5),
    A.Normalize(mean=mean, std=std),
    ToTensorV2()
])

class CIFAR10Albumentations(CIFAR10):
    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]
        augmented = albumentations_transform(image=image)
        image = augmented["image"]
        return image, label
