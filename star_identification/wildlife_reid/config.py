"""
Configuration for Wildlife ReID-10k training.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from pathlib import Path
import json


@dataclass
class SplitConfig:
    """Configuration for dataset splitting."""
    
    # Split strategy: "original" | "time_aware" | "cluster_aware" | "random"
    strategy: str = "original"
    
    # Train ratio (used for random/cluster splits when re-splitting)
    train_ratio: float = 0.8
    
    # Minimum images per identity to include
    min_images_per_identity: int = 2
    
    # Minimum identities with required images
    min_identities: int = 10
    
    # For time-aware splitting
    time_train_ratio: float = 0.7  # Earlier X% of dates go to train
    
    # For cluster-aware splitting  
    cluster_similarity_threshold: float = 0.85
    
    # Random seed for reproducibility
    seed: int = 42


@dataclass
class FilterConfig:
    """Configuration for filtering datasets/samples."""
    
    # Include only these datasets (empty = all)
    include_datasets: List[str] = field(default_factory=list)
    
    # Exclude these datasets
    exclude_datasets: List[str] = field(default_factory=list)
    
    # Include only these species (empty = all)
    include_species: List[str] = field(default_factory=list)
    
    # Exclude these species
    exclude_species: List[str] = field(default_factory=list)
    
    # Require orientation field
    require_orientation: bool = False
    
    # Require date field  
    require_date: bool = False


@dataclass
class Wildlife10kConfig:
    """Main configuration for Wildlife10k training."""
    
    # Path to wildlifeReID folder
    data_root: str = "./wildlifeReID"
    
    # Split configuration
    split: SplitConfig = field(default_factory=SplitConfig)
    
    # Filter configuration
    filter: FilterConfig = field(default_factory=FilterConfig)
    
    # Image size
    image_size: int = 384
    
    # Training parameters
    batch_size: int = 32
    num_instances: int = 4  # K in P-K sampling
    
    # Model
    model_name: str = "microsoft/swinv2-small-patch4-window16-256"
    embedding_dim: int = 512
    dropout: float = 0.1
    pretrained: bool = True
    
    # Training
    num_epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    
    # Loss weights
    circle_weight: float = 0.7
    triplet_weight: float = 0.3
    
    # Hardware
    device: str = "cuda"
    num_workers: int = 4
    pin_memory: bool = True
    use_amp: bool = True
    
    # Experiment tracking
    experiment_name: str = "wildlife10k_baseline"
    checkpoint_dir: str = "./checkpoints/wildlife10k"
    log_dir: str = "./logs/wildlife10k"
    
    seed: int = 42
    
    def __post_init__(self):
        """Validate and setup directories."""
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        
        # Convert nested dicts to dataclasses if needed
        if isinstance(self.split, dict):
            self.split = SplitConfig(**self.split)
        if isinstance(self.filter, dict):
            self.filter = FilterConfig(**self.filter)
    
    def save(self, path: Path):
        """Save config to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = {
            'data_root': self.data_root,
            'split': {
                'strategy': self.split.strategy,
                'train_ratio': self.split.train_ratio,
                'min_images_per_identity': self.split.min_images_per_identity,
                'min_identities': self.split.min_identities,
                'time_train_ratio': self.split.time_train_ratio,
                'cluster_similarity_threshold': self.split.cluster_similarity_threshold,
                'seed': self.split.seed,
            },
            'filter': {
                'include_datasets': self.filter.include_datasets,
                'exclude_datasets': self.filter.exclude_datasets,
                'include_species': self.filter.include_species,
                'exclude_species': self.filter.exclude_species,
                'require_orientation': self.filter.require_orientation,
                'require_date': self.filter.require_date,
            },
            'image_size': self.image_size,
            'batch_size': self.batch_size,
            'num_instances': self.num_instances,
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
            'dropout': self.dropout,
            'num_epochs': self.num_epochs,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'warmup_epochs': self.warmup_epochs,
            'circle_weight': self.circle_weight,
            'triplet_weight': self.triplet_weight,
            'device': self.device,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
            'use_amp': self.use_amp,
            'experiment_name': self.experiment_name,
            'checkpoint_dir': self.checkpoint_dir,
            'log_dir': self.log_dir,
            'seed': self.seed,
        }
        
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> 'Wildlife10kConfig':
        """Load config from JSON."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)
    
    def summary(self) -> str:
        """Return a brief summary string."""
        return (
            f"Wildlife10k | Split: {self.split.strategy} | "
            f"Image: {self.image_size}px | Batch: {self.batch_size}x{self.num_instances} | "
            f"LR: {self.learning_rate}"
        )

