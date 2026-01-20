"""
Configuration dataclasses for identity verification.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Tuple


@dataclass
class BackboneConfig:
    """Configuration for the shared backbone."""
    
    # Backbone architecture (must match checkpoint)
    name: str = "convnext-small"
    
    # Path to pre-trained embedding model checkpoint
    checkpoint_path: Optional[str] = None
    
    # Whether to freeze backbone during training
    freeze: bool = False
    
    # Which layer to extract features from (None = last feature map)
    feature_layer: Optional[str] = None
    
    # Image size (must match backbone training)
    image_size: int = 224


@dataclass
class CrossAttentionConfig:
    """Configuration for the cross-attention module."""
    
    # Feature dimension from backbone (ConvNeXt-small: 768)
    feature_dim: int = 768
    
    # Hidden dimension for cross-attention
    hidden_dim: int = 256
    
    # Number of cross-attention layers
    num_layers: int = 2
    
    # Number of attention heads
    num_heads: int = 8
    
    # Dropout rate
    dropout: float = 0.1
    
    # Whether to use learnable position embeddings
    use_pos_embed: bool = True


@dataclass
class VerificationConfig:
    """Full model configuration."""
    
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    cross_attention: CrossAttentionConfig = field(default_factory=CrossAttentionConfig)
    
    # Classification head
    classifier_hidden: int = 256
    classifier_dropout: float = 0.3
    
    def __post_init__(self):
        if isinstance(self.backbone, dict):
            self.backbone = BackboneConfig(**self.backbone)
        if isinstance(self.cross_attention, dict):
            self.cross_attention = CrossAttentionConfig(**self.cross_attention)
    
    @property
    def image_size(self) -> int:
        """Get image size from backbone config."""
        return self.backbone.image_size
    
    def to_dict(self) -> dict:
        """Convert config to dictionary for serialization."""
        from dataclasses import asdict
        return asdict(self)


@dataclass
class TrainingConfig:
    """Training configuration."""
    
    # Data paths
    wildlife_root: str = "./wildlifeReID_resized"
    star_dataset_root: str = "./star_dataset_resized"
    
    # Output
    checkpoint_dir: str = "./checkpoints/verification"
    log_dir: str = "./logs/verification"
    
    # Training
    batch_size: int = 32  # Number of pairs per batch
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    
    # Pair sampling
    pairs_per_epoch: int = 50000  # Total pairs to sample per epoch
    positive_ratio: float = 0.5   # Fraction of positive pairs
    
    # Scheduler
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    
    # Hardware
    device: str = "cuda"
    num_workers: int = 8
    use_amp: bool = True
    
    # Validation
    val_every: int = 5
    val_pairs: int = 10000  # Pairs to use for validation
    
    # Reproducibility
    seed: int = 42
    
    # Resume
    resume_checkpoint: Optional[str] = None
    
    def __post_init__(self):
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)


@dataclass 
class PretrainConfig(TrainingConfig):
    """Pre-training on Wildlife10k."""
    
    checkpoint_dir: str = "./checkpoints/verification/pretrain"
    
    # Split strategy for wildlife data
    split_strategy: str = "recommended"
    
    # Exclude sea stars for fair evaluation
    exclude_datasets: List[str] = field(default_factory=lambda: ["SeaStarReID2023"])


@dataclass
class FinetuneConfig(TrainingConfig):
    """Fine-tuning on star dataset."""
    
    checkpoint_dir: str = "./checkpoints/verification/finetune"
    
    # Pre-trained verification checkpoint to load
    pretrain_checkpoint: Optional[str] = None
    
    # Lower learning rate for fine-tuning
    learning_rate: float = 1e-5
    
    # Smaller epoch count
    num_epochs: int = 50
    pairs_per_epoch: int = 10000





