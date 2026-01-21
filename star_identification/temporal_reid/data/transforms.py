"""
Image transforms for training and validation using Albumentations.

Optimized for underwater sea star re-identification with:
- Perspective and affine transforms (different camera angles)
- Underwater-specific color augmentations (blue shift, turbidity)
- Motion blur (common in underwater footage)
- CLAHE for contrast enhancement
- Noise and compression artifacts
"""
import numpy as np
from PIL import Image
from typing import Callable

import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torchvision.transforms import RandomErasing


class AlbumentationsTransform:
    """
    Wrapper that applies Albumentations transforms to PIL Images.
    
    Handles PIL -> numpy -> tensor conversion and optionally applies
    RandomErasing on the output tensor.
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
        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Handle RGBA images (convert to RGB)
        if image.ndim == 3 and image.shape[2] == 4:
            image = image[:, :, :3]
        
        # Apply albumentations pipeline
        augmented = self.transform(image=image)['image']
        
        # Apply random erasing on tensor if configured
        if self.random_erasing is not None:
            augmented = self.random_erasing(augmented)
        
        return augmented


def create_train_transform(config) -> Callable:
    """
    Create training transform with comprehensive augmentation.
    
    Uses Albumentations for better performance and more augmentation options,
    specifically tuned for underwater sea star imagery.
    
    Args:
        config: Config object with augmentation settings
    
    Returns:
        Callable transform that accepts PIL Image and returns tensor
    """
    image_size = config.get_image_size()
    aug = config.augmentation
    
    # Build the augmentation pipeline
    transform = A.Compose([
        # Resize with slack for random crop (1.3x = up to 30% zoom variation)
        A.LongestMaxSize(max_size=int(image_size * 1.3)),
        A.PadIfNeeded(
            min_height=int(image_size * 1.3),
            min_width=int(image_size * 1.3),
            border_mode=0,  # cv2.BORDER_CONSTANT
            fill=(0, 0, 0)
        ),
        A.RandomCrop(height=image_size, width=image_size),
        
        # === Geometric Transforms ===
        # Basic flips
        A.HorizontalFlip(p=aug.horizontal_flip_prob),
        A.VerticalFlip(p=aug.vertical_flip_prob),
        A.RandomRotate90(p=aug.rotate90_prob),
        
        # Affine transforms (scale, translate, rotate, shear)
        A.Affine(
            scale=(aug.affine_scale[0], aug.affine_scale[1]),
            translate_percent={
                'x': (-aug.affine_translate, aug.affine_translate),
                'y': (-aug.affine_translate, aug.affine_translate)
            },
            rotate=(-aug.affine_rotate, aug.affine_rotate),
            shear=(-aug.affine_shear, aug.affine_shear),
            border_mode=0,  # cv2.BORDER_CONSTANT
            p=aug.affine_prob
        ),
        
        # Perspective transform (viewpoint changes)
        A.Perspective(
            scale=(aug.perspective_scale[0], aug.perspective_scale[1]),
            p=aug.perspective_prob
        ),
        
        # === Optical Distortions (lens effects) ===
        A.OneOf([
            A.OpticalDistortion(
                distort_limit=aug.optical_distortion_limit,
                p=1.0
            ),
            A.GridDistortion(
                num_steps=aug.grid_distortion_steps,
                distort_limit=0.3,
                p=1.0
            ),
            A.ElasticTransform(
                alpha=aug.elastic_alpha,
                sigma=aug.elastic_sigma,
                p=1.0
            ),
        ], p=aug.optical_distortion_prob),
        
        # === Color Transforms (underwater-optimized) ===
        # OneOf: different color augmentation strategies
        A.OneOf([
            A.ColorJitter(
                brightness=aug.brightness,
                contrast=aug.contrast,
                saturation=aug.saturation,
                hue=aug.hue,
                p=1.0
            ),
            # RGBShift with extra blue for underwater
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
        
        # Contrast enhancement (CLAHE is great for underwater)
        A.OneOf([
            A.CLAHE(clip_limit=aug.clahe_clip_limit, tile_grid_size=(8, 8), p=1.0),
            A.Equalize(p=1.0),
        ], p=aug.clahe_prob),
        
        # Grayscale (reduced probability - color patterns matter!)
        A.ToGray(p=aug.grayscale_prob),
        
        # === Blur and Sharpness ===
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, aug.blur_limit), p=1.0),
            A.MotionBlur(blur_limit=aug.motion_blur_limit, p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
        ], p=aug.blur_prob),
        
        A.OneOf([
            A.Sharpen(alpha=(0.2, 0.4), lightness=(0.5, 1.0), p=1.0),
            A.UnsharpMask(blur_limit=(3, 7), p=1.0),
        ], p=aug.sharpen_prob),
        
        # === Spackling / Dropouts ===
        # Spatter: simulates water drops, mud, debris (common underwater)
        A.Spatter(
            mean=(0.5, 0.6),
            std=(0.1, 0.2),
            gauss_sigma=(2, 3),
            cutout_threshold=(0.6, 0.7),
            intensity=(0.3, 0.5),
            mode='mud',
            p=aug.spatter_prob
        ),
        
        # Coarse dropout: random rectangular holes
        A.CoarseDropout(
            num_holes_range=(1, aug.coarse_dropout_max_holes),
            hole_height_range=(int(image_size * 0.01), int(image_size * aug.coarse_dropout_max_size)),
            hole_width_range=(int(image_size * 0.01), int(image_size * aug.coarse_dropout_max_size)),
            fill=0,
            p=aug.coarse_dropout_prob
        ),
        
        # Pixel dropout: random individual pixels removed
        A.PixelDropout(
            dropout_prob=aug.pixel_dropout_rate,
            p=aug.pixel_dropout_prob
        ),
        
        # === Noise and Quality Degradation ===
        A.OneOf([
            A.GaussNoise(std_range=(0.02, 0.08), p=1.0),  # std as fraction of [0,1] range
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
        
        # === Normalize and Convert to Tensor ===
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


def create_val_transform(config) -> Callable:
    """
    Create validation/test transform (no augmentation).
    
    Args:
        config: Config object
    
    Returns:
        Callable transform that accepts PIL Image and returns tensor
    """
    image_size = config.get_image_size()
    
    transform = A.Compose([
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(
            min_height=image_size,
            min_width=image_size,
            border_mode=0,
            fill=(0, 0, 0)
        ),
        A.CenterCrop(height=image_size, width=image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(),
    ])
    
    return AlbumentationsTransform(transform=transform, random_erasing_prob=0.0)


# === Legacy torchvision transforms (kept for compatibility) ===

def create_train_transform_legacy(config):
    """
    Legacy training transform using torchvision.
    Use create_train_transform() instead for better augmentation.
    """
    from torchvision import transforms
    
    image_size = config.get_image_size()
    aug = config.augmentation
    
    transform_list = [
        transforms.Resize((int(image_size * 1.1), int(image_size * 1.1))),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(p=aug.horizontal_flip_prob),
        transforms.RandomVerticalFlip(p=aug.vertical_flip_prob),
        transforms.RandomRotation(degrees=aug.affine_rotate),
    ]
    
    if aug.color_augment_prob > 0:
        transform_list.append(
            transforms.RandomApply([
                transforms.ColorJitter(
                    brightness=aug.brightness,
                    contrast=aug.contrast,
                    saturation=aug.saturation,
                    hue=aug.hue
                )
            ], p=aug.color_augment_prob)
        )
    
    if aug.grayscale_prob > 0:
        transform_list.append(transforms.RandomGrayscale(p=aug.grayscale_prob))
    
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    
    if aug.random_erasing_prob > 0:
        transform_list.append(
            RandomErasing(
                p=aug.random_erasing_prob,
                scale=aug.random_erasing_scale,
                ratio=(0.3, 3.3),
                value='random'
            )
        )
    
    return transforms.Compose(transform_list)


def create_val_transform_legacy(config):
    """
    Legacy validation transform using torchvision.
    Use create_val_transform() instead.
    """
    from torchvision import transforms
    
    image_size = config.get_image_size()
    
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
