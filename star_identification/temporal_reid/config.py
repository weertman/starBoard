"""
Configuration for temporal re-identification training.
"""
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import os


@dataclass
class AugmentationConfig:
    """Data augmentation settings for underwater sea star re-identification."""
    
    # Geometric - basic flips
    horizontal_flip_prob: float = 0.5
    vertical_flip_prob: float = 0.5
    rotate90_prob: float = 0.5
    
    # Geometric - affine transforms
    affine_prob: float = 0.5
    affine_scale: Tuple[float, float] = (0.75, 1.25)  # ±25% scale variation
    affine_translate: float = 0.1
    affine_rotate: float = 30.0
    affine_shear: float = 10.0
    
    # Geometric - perspective
    perspective_prob: float = 0.3
    perspective_scale: Tuple[float, float] = (0.02, 0.08)
    
    # Geometric - optical distortions (lens effects)
    optical_distortion_prob: float = 0.3
    optical_distortion_limit: float = 0.3
    grid_distortion_prob: float = 0.2
    grid_distortion_steps: int = 5
    elastic_prob: float = 0.15
    elastic_alpha: float = 1.0
    elastic_sigma: float = 50.0
    
    # Color - jitter options (OneOf selection)
    color_augment_prob: float = 0.5
    brightness: float = 0.2
    contrast: float = 0.2
    saturation: float = 0.3
    hue: float = 0.15
    rgb_shift_limit: int = 20
    blue_shift_extra: int = 10  # Extra blue shift for underwater
    
    # Color - contrast enhancement
    clahe_prob: float = 0.2
    clahe_clip_limit: float = 4.0
    
    # Grayscale (reduced - color patterns matter for stars!)
    grayscale_prob: float = 0.1
    
    # Blur/sharpness
    blur_prob: float = 0.15
    blur_limit: int = 7
    motion_blur_limit: int = 5
    sharpen_prob: float = 0.2
    
    # Noise/quality degradation
    noise_prob: float = 0.2
    gauss_noise_var: Tuple[float, float] = (10.0, 50.0)
    iso_noise_intensity: Tuple[float, float] = (0.1, 0.3)
    compression_quality: Tuple[int, int] = (70, 95)
    
    # Spackling / dropouts
    spatter_prob: float = 0.15  # Water drops, mud, debris simulation
    coarse_dropout_prob: float = 0.2  # Random rectangular holes
    coarse_dropout_max_holes: int = 12
    coarse_dropout_max_size: float = 0.05  # As fraction of image
    pixel_dropout_prob: float = 0.1  # Random pixel removal
    pixel_dropout_rate: float = 0.02
    
    # Erasing (applied on tensor)
    random_erasing_prob: float = 0.2
    random_erasing_scale: Tuple[float, float] = (0.02, 0.2)


@dataclass 
class LossConfig:
    """Loss function settings."""
    
    # Loss weights
    circle_weight: float = 0.7
    triplet_weight: float = 0.3
    arcface_weight: float = 0.0  # Disabled by default for negative-only setup
    
    # Loss parameters
    circle_margin: float = 0.25
    circle_scale: float = 64.0
    triplet_margin: float = 0.3
    arcface_margin: float = 0.5
    arcface_scale: float = 64.0


@dataclass
class TemporalSplitConfig:
    """Settings for temporal train/test splitting."""
    
    # Split ratio (fraction of outings for training)
    train_outing_ratio: float = 0.8
    
    # Minimum outings required to be a "multi-outing" identity
    min_outings_for_eval: int = 2
    
    # How to handle identities without dates
    undated_policy: str = "train_only"  # "train_only" | "exclude"
    
    # Random seed for any random decisions in splitting
    seed: int = 42


@dataclass
class Config:
    """Main configuration."""
    
    # Paths
    star_dataset_root: str = "./star_dataset"
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    
    # Model (small backbone + large images for best detail/efficiency tradeoff)
    model_name: str = "microsoft/swinv2-small-patch4-window16-256"
    embedding_dim: int = 512
    dropout: float = 0.1
    pretrained: bool = True
    image_size: Optional[int] = 384  # 384px with small model; None = auto-detect from model name
    
    # Training
    num_epochs: int = 100
    batch_size: int = 32
    num_instances: int = 4  # K in P-K sampling
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    
    # Validation
    val_every_n_epochs: int = 5
    
    # Hardware
    device: str = "cuda"
    num_workers: int = 4
    pin_memory: bool = True
    use_amp: bool = True
    
    # Experiment
    experiment_name: str = "temporal_reid"
    seed: int = 42
    
    # Sub-configs
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    temporal_split: TemporalSplitConfig = field(default_factory=TemporalSplitConfig)
    
    def __post_init__(self):
        """Create directories and set derived values."""
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        
        # Auto-detect num_workers for Windows
        if self.num_workers is None:
            if os.name == 'nt':
                self.num_workers = min(4, os.cpu_count() or 1)
            else:
                self.num_workers = min(8, os.cpu_count() or 1)
    
    def get_image_size(self) -> int:
        """Get appropriate image size for the model."""
        if self.image_size is not None:
            return self.image_size
        # Fallback: auto-detect from model name
        name_lower = self.model_name.lower()
        if "384" in name_lower:
            return 384
        elif "256" in name_lower:
            return 256
        elif "224" in name_lower:
            return 224
        return 384  # Default
    
    def save(self, path: Path):
        """Save config to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = asdict(self)
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> 'Config':
        """Load config from JSON."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Handle nested dataclasses
        if 'augmentation' in data and isinstance(data['augmentation'], dict):
            data['augmentation'] = AugmentationConfig(**data['augmentation'])
        if 'loss' in data and isinstance(data['loss'], dict):
            data['loss'] = LossConfig(**data['loss'])
        if 'temporal_split' in data and isinstance(data['temporal_split'], dict):
            data['temporal_split'] = TemporalSplitConfig(**data['temporal_split'])
        
        return cls(**data)
    
    def summary(self) -> str:
        """Return a summary string."""
        return (
            f"Model: {self.model_name.split('/')[-1]} | "
            f"Image: {self.get_image_size()}px | "
            f"Embed: {self.embedding_dim} | "
            f"Batch: {self.batch_size}x{self.num_instances} | "
            f"LR: {self.learning_rate}"
        )

