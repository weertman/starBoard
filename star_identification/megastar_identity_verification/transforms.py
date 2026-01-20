"""
Transform wrappers for verification training.

Re-exports existing augmentation pipelines from megastarid for consistency,
while keeping this subproject compartmentalized.
"""
from typing import Tuple
import torchvision.transforms as T


def get_train_transforms(image_size: int = 224) -> T.Compose:
    """
    Get training transforms with augmentation.
    
    Matches the augmentation used in embedding model training for consistency.
    
    Args:
        image_size: Target image size
        
    Returns:
        Composed transform
    """
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        T.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.1,
            hue=0.05,
        ),
        T.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1),
        ),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def get_test_transforms(image_size: int = 224) -> T.Compose:
    """
    Get test/validation transforms (no augmentation).
    
    Args:
        image_size: Target image size
        
    Returns:
        Composed transform
    """
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def get_strong_train_transforms(image_size: int = 224) -> T.Compose:
    """
    Get stronger training transforms for more robust learning.
    
    Use when you want more aggressive augmentation.
    
    Args:
        image_size: Target image size
        
    Returns:
        Composed transform
    """
    return T.Compose([
        T.Resize((int(image_size * 1.1), int(image_size * 1.1))),
        T.RandomCrop(image_size),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.2),  # Sea stars can be any orientation
        T.RandomRotation(degrees=30),
        T.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.2,
            hue=0.1,
        ),
        T.RandomAffine(
            degrees=0,
            translate=(0.15, 0.15),
            scale=(0.85, 1.15),
        ),
        T.RandomPerspective(distortion_scale=0.2, p=0.3),
        T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        T.RandomErasing(p=0.2, scale=(0.02, 0.15)),
    ])


# Try to import from existing megastarid transforms if available
try:
    from megastarid.transforms import (
        get_wildlife_train_transforms as _megastarid_train,
        get_wildlife_test_transforms as _megastarid_test,
    )
    
    def get_megastarid_train_transforms(image_size: int = 224) -> T.Compose:
        """Use exact same transforms as embedding model training."""
        return _megastarid_train(image_size)
    
    def get_megastarid_test_transforms(image_size: int = 224) -> T.Compose:
        """Use exact same transforms as embedding model evaluation."""
        return _megastarid_test(image_size)
    
    MEGASTARID_TRANSFORMS_AVAILABLE = True
    
except ImportError:
    MEGASTARID_TRANSFORMS_AVAILABLE = False
    get_megastarid_train_transforms = get_train_transforms
    get_megastarid_test_transforms = get_test_transforms






