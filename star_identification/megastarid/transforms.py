"""
Transform utilities for MegaStarID.

Provides two augmentation strategies:
- Wildlife10k: Standard torchvision transforms (good for diverse species)
- star_dataset: Sophisticated Albumentations transforms (tuned for underwater sea stars)
"""
from typing import Callable, Tuple
from dataclasses import dataclass

import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms import RandomErasing

import albumentations as A
from albumentations.pytorch import ToTensorV2


# =============================================================================
# Configuration for star_dataset augmentations
# =============================================================================

@dataclass
class StarAugmentationConfig:
    """Augmentation settings optimized for underwater sea star imagery."""
    
    # Geometric - basic flips
    horizontal_flip_prob: float = 0.5
    vertical_flip_prob: float = 0.5
    rotate90_prob: float = 0.5
    
    # Geometric - affine transforms
    affine_prob: float = 0.5
    affine_scale: Tuple[float, float] = (0.75, 1.75)  # More zoom-in range for scale variation
    affine_translate: float = 0.1
    affine_rotate: float = 45.0  # Increased rotation range
    affine_shear: float = 10.0
    
    # Geometric - perspective
    perspective_prob: float = 0.3
    perspective_scale: Tuple[float, float] = (0.02, 0.08)
    
    # Geometric - optical distortions
    optical_distortion_prob: float = 0.3
    optical_distortion_limit: float = 0.3
    grid_distortion_steps: int = 5
    elastic_alpha: float = 1.0
    elastic_sigma: float = 50.0
    
    # Color (underwater-optimized)
    color_augment_prob: float = 0.5
    brightness: float = 0.2
    contrast: float = 0.2
    saturation: float = 0.3
    hue: float = 0.3  # Increased for underwater lighting variation
    rgb_shift_limit: int = 20
    blue_shift_extra: int = 10  # Extra blue for underwater
    
    # Contrast enhancement
    clahe_prob: float = 0.2
    clahe_clip_limit: float = 4.0
    
    # Grayscale (low - color patterns matter)
    grayscale_prob: float = 0.1
    
    # Blur/sharpness
    blur_prob: float = 0.15
    blur_limit: int = 7
    motion_blur_limit: int = 5
    sharpen_prob: float = 0.2
    
    # Noise/quality
    noise_prob: float = 0.2
    iso_noise_intensity: Tuple[float, float] = (0.1, 0.3)
    compression_quality: Tuple[int, int] = (70, 95)
    
    # Dropouts
    spatter_prob: float = 0.15
    coarse_dropout_prob: float = 0.2
    coarse_dropout_max_holes: int = 12
    coarse_dropout_max_size: float = 0.05
    pixel_dropout_prob: float = 0.1
    pixel_dropout_rate: float = 0.02
    
    # Random erasing
    random_erasing_prob: float = 0.2


# =============================================================================
# Albumentations wrapper for star_dataset
# =============================================================================

class AlbumentationsTransform:
    """
    Wrapper that applies Albumentations transforms to PIL Images.
    """
    
    def __init__(self, transform: A.Compose, random_erasing_prob: float = 0.0):
        self.transform = transform
        self.random_erasing = None
        if random_erasing_prob > 0:
            self.random_erasing = RandomErasing(
                p=random_erasing_prob,
                scale=(0.02, 0.2),
                ratio=(0.3, 3.3),
                value='random'
            )
    
    def __call__(self, image):
        # Convert PIL to numpy
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Handle RGBA
        if image.ndim == 3 and image.shape[2] == 4:
            image = image[:, :, :3]
        
        # Apply albumentations
        augmented = self.transform(image=image)['image']
        
        # Random erasing on tensor
        if self.random_erasing is not None:
            augmented = self.random_erasing(augmented)
        
        return augmented


# =============================================================================
# Star dataset transforms (Albumentations - sophisticated)
# =============================================================================

def get_star_train_transforms(
    image_size: int = 384,
    aug_config: StarAugmentationConfig = None,
) -> Callable:
    """
    Get training transforms for star_dataset using Albumentations.
    
    Includes underwater-specific augmentations:
    - Perspective and affine transforms
    - Blue-shifted color augmentation
    - Motion blur, CLAHE
    - Noise and compression artifacts
    """
    aug = aug_config or StarAugmentationConfig()
    
    transform = A.Compose([
        # Resize with slack for random crop
        A.LongestMaxSize(max_size=int(image_size * 1.3)),
        A.PadIfNeeded(
            min_height=int(image_size * 1.3),
            min_width=int(image_size * 1.3),
            border_mode=0,
            value=(0, 0, 0)
        ),
        A.RandomCrop(height=image_size, width=image_size),
        
        # === Geometric ===
        A.HorizontalFlip(p=aug.horizontal_flip_prob),
        A.VerticalFlip(p=aug.vertical_flip_prob),
        A.RandomRotate90(p=aug.rotate90_prob),
        
        A.Affine(
            scale=(aug.affine_scale[0], aug.affine_scale[1]),
            translate_percent={
                'x': (-aug.affine_translate, aug.affine_translate),
                'y': (-aug.affine_translate, aug.affine_translate)
            },
            rotate=(-aug.affine_rotate, aug.affine_rotate),
            shear=(-aug.affine_shear, aug.affine_shear),
            border_mode=0,
            p=aug.affine_prob
        ),
        
        A.Perspective(
            scale=(aug.perspective_scale[0], aug.perspective_scale[1]),
            p=aug.perspective_prob
        ),
        
        # === Optical Distortions ===
        A.OneOf([
            A.OpticalDistortion(distort_limit=aug.optical_distortion_limit, p=1.0),
            A.GridDistortion(num_steps=aug.grid_distortion_steps, distort_limit=0.3, p=1.0),
            A.ElasticTransform(alpha=aug.elastic_alpha, sigma=aug.elastic_sigma, p=1.0),
        ], p=aug.optical_distortion_prob),
        
        # === Color (underwater-optimized) ===
        A.OneOf([
            A.ColorJitter(
                brightness=aug.brightness,
                contrast=aug.contrast,
                saturation=aug.saturation,
                hue=aug.hue,
                p=1.0
            ),
            A.RGBShift(
                r_shift_limit=aug.rgb_shift_limit,
                g_shift_limit=aug.rgb_shift_limit,
                b_shift_limit=aug.rgb_shift_limit + aug.blue_shift_extra,
                p=1.0
            ),
            A.HueSaturationValue(
                hue_shift_limit=int(aug.hue * 100),
                sat_shift_limit=int(aug.saturation * 100),
                val_shift_limit=int(aug.brightness * 100),
                p=1.0
            ),
        ], p=aug.color_augment_prob),
        
        # Contrast enhancement
        A.OneOf([
            A.CLAHE(clip_limit=aug.clahe_clip_limit, tile_grid_size=(8, 8), p=1.0),
            A.Equalize(p=1.0),
        ], p=aug.clahe_prob),
        
        A.ToGray(p=aug.grayscale_prob),
        
        # === Blur/Sharpness ===
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, aug.blur_limit), p=1.0),
            A.MotionBlur(blur_limit=aug.motion_blur_limit, p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
        ], p=aug.blur_prob),
        
        A.OneOf([
            A.Sharpen(alpha=(0.2, 0.4), lightness=(0.5, 1.0), p=1.0),
            A.UnsharpMask(blur_limit=(3, 7), p=1.0),
        ], p=aug.sharpen_prob),
        
        # === Dropouts ===
        A.Spatter(
            mean=(0.5, 0.6),
            std=(0.1, 0.2),
            gauss_sigma=(2, 3),
            cutout_threshold=(0.6, 0.7),
            intensity=(0.3, 0.5),
            mode='mud',
            p=aug.spatter_prob
        ),
        
        A.CoarseDropout(
            num_holes_range=(1, aug.coarse_dropout_max_holes),
            hole_height_range=(int(image_size * 0.01), int(image_size * aug.coarse_dropout_max_size)),
            hole_width_range=(int(image_size * 0.01), int(image_size * aug.coarse_dropout_max_size)),
            fill=0,
            p=aug.coarse_dropout_prob
        ),
        
        A.PixelDropout(
            dropout_prob=aug.pixel_dropout_rate,
            p=aug.pixel_dropout_prob
        ),
        
        # === Noise/Quality ===
        A.OneOf([
            A.GaussNoise(std_range=(0.02, 0.08), p=1.0),
            A.ISONoise(
                color_shift=(0.01, 0.05),
                intensity=aug.iso_noise_intensity,
                p=1.0
            ),
            A.ImageCompression(
                quality_range=(aug.compression_quality[0], aug.compression_quality[1]),
                p=1.0
            ),
        ], p=aug.noise_prob),
        
        # === Normalize ===
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(),
    ])
    
    return AlbumentationsTransform(
        transform=transform,
        random_erasing_prob=aug.random_erasing_prob
    )


def get_star_test_transforms(image_size: int = 384) -> Callable:
    """Get test transforms for star_dataset (no augmentation)."""
    transform = A.Compose([
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(
            min_height=image_size,
            min_width=image_size,
            border_mode=0,
            value=(0, 0, 0)
        ),
        A.CenterCrop(height=image_size, width=image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(),
    ])
    
    return AlbumentationsTransform(transform=transform, random_erasing_prob=0.0)


# =============================================================================
# Wildlife10k transforms (torchvision - standard)
# =============================================================================

def get_wildlife_train_transforms(image_size: int = 384) -> transforms.Compose:
    """
    Get training transforms for Wildlife10k using torchvision.
    
    Standard augmentations suitable for diverse animal species.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(
            image_size,
            scale=(0.7, 1.0),
            ratio=(0.8, 1.2),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1,
        ),
        transforms.RandomGrayscale(p=0.1),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
        ], p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
    ])


def get_wildlife_test_transforms(image_size: int = 384) -> transforms.Compose:
    """Get test transforms for Wildlife10k (no augmentation)."""
    return transforms.Compose([
        transforms.Resize(
            (image_size, image_size),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


