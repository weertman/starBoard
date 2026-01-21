"""
Transform utilities for Wildlife ReID.
"""
from typing import Tuple, Optional
import torch
from torchvision import transforms


def get_train_transforms(
    image_size: int = 384,
    normalize: bool = True,
) -> transforms.Compose:
    """
    Get training transforms with augmentation.
    
    Args:
        image_size: Target image size
        normalize: Whether to apply ImageNet normalization
        
    Returns:
        Composed transforms
    """
    transform_list = [
        # Resize with some random cropping
        transforms.RandomResizedCrop(
            image_size,
            scale=(0.7, 1.0),
            ratio=(0.8, 1.2),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        
        # Geometric augmentations
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=30),
        
        # Color augmentations
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1,
        ),
        
        # Random grayscale (many wildlife datasets work well with patterns)
        transforms.RandomGrayscale(p=0.1),
        
        # Blur (simulates camera/motion blur)
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
        ], p=0.1),
        
        # To tensor
        transforms.ToTensor(),
    ]
    
    if normalize:
        transform_list.append(
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        )
    
    # Random erasing (after normalization)
    transform_list.append(
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15))
    )
    
    return transforms.Compose(transform_list)


def get_test_transforms(
    image_size: int = 384,
    normalize: bool = True,
) -> transforms.Compose:
    """
    Get test/validation transforms (no augmentation).
    
    Args:
        image_size: Target image size
        normalize: Whether to apply ImageNet normalization
        
    Returns:
        Composed transforms
    """
    transform_list = [
        transforms.Resize(
            (image_size, image_size),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.ToTensor(),
    ]
    
    if normalize:
        transform_list.append(
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        )
    
    return transforms.Compose(transform_list)


def get_tta_transforms(
    image_size: int = 384,
    normalize: bool = True,
) -> Tuple[transforms.Compose, int]:
    """
    Get test-time augmentation transforms.
    
    Returns multiple augmented views for averaging predictions.
    
    Args:
        image_size: Target image size
        normalize: Whether to apply ImageNet normalization
        
    Returns:
        Tuple of (transform, num_augmentations)
    """
    # Base transform
    base_transforms = [
        transforms.Resize(
            (image_size, image_size),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
    ]
    
    post_transforms = [transforms.ToTensor()]
    if normalize:
        post_transforms.append(
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        )
    
    # TTA augmentations (4 views total)
    class TTATransform:
        """Apply multiple augmented views."""
        
        def __init__(self, base, post, image_size):
            self.base = transforms.Compose(base)
            self.post = transforms.Compose(post)
            self.image_size = image_size
        
        def __call__(self, img):
            """Return list of augmented tensors."""
            views = []
            
            # Original
            base_img = self.base(img)
            views.append(self.post(base_img))
            
            # Horizontal flip
            views.append(self.post(transforms.functional.hflip(base_img)))
            
            # Vertical flip  
            views.append(self.post(transforms.functional.vflip(base_img)))
            
            # Both flips
            views.append(self.post(
                transforms.functional.vflip(
                    transforms.functional.hflip(base_img)
                )
            ))
            
            return torch.stack(views)
    
    return TTATransform(base_transforms, post_transforms, image_size), 4


