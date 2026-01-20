"""
Configuration for MegaStarID unified training.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from pathlib import Path
import json


@dataclass
class ModelConfig:
    """Model architecture settings."""
    
    # Backbone selection: "swinv2-tiny", "densenet121", "densenet169"
    backbone: str = "swinv2-tiny"
    
    # Legacy field (for backward compatibility)
    name: str = "microsoft/swinv2-tiny-patch4-window8-256"
    
    # Feature extraction
    use_multiscale: bool = True
    multiscale_stages: Tuple[int, ...] = (2, 3, 4)
    
    # Embedding head
    embedding_dim: int = 512
    embedding_head_depth: int = 3  # 2 or 3 layers
    use_residual_head: bool = False
    dropout: float = 0.1
    
    # Pooling
    use_attention_pooling: bool = False
    
    # BNNeck (separate BN for training/inference)
    use_bnneck: bool = True
    
    # Input
    image_size: int = 384
    pretrained: bool = True
    
    # Inference enhancements
    use_tta: bool = False  # Test-time augmentation
    use_reranking: bool = False  # k-reciprocal re-ranking


@dataclass
class LossConfig:
    """Loss function settings."""
    circle_weight: float = 0.7
    triplet_weight: float = 0.3
    circle_margin: float = 0.25
    circle_scale: float = 64.0
    triplet_margin: float = 0.3


@dataclass
class MegaStarConfig:
    """Base configuration shared across all training modes."""
    
    # Paths - use resized versions if available for faster loading
    # Run: python -m wildlife_reid_utils.preprocess_images --size 620
    wildlife_root: str = "./wildlifeReID_resized"  # or "./wildlifeReID" for originals
    star_dataset_root: str = "./star_dataset_resized"  # or "./star_dataset" for originals
    checkpoint_dir: str = "./checkpoints/megastarid"
    log_dir: str = "./logs/megastarid"
    
    # Model
    model: ModelConfig = field(default_factory=ModelConfig)
    
    # Loss
    loss: LossConfig = field(default_factory=LossConfig)
    
    # Training basics
    batch_size: int = 32
    num_instances: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    warmup_ratio: Optional[float] = None  # If set, overrides warmup_epochs as fraction of num_epochs
    
    # Gradient clipping
    grad_clip_norm: Optional[float] = 1.0  # Max gradient norm, None to disable
    
    # Layer-wise learning rate decay (for transformers)
    use_llrd: bool = False                 # Enable layer-wise LR decay
    llrd_decay: float = 0.75               # Decay factor per layer (deeper = lower LR)
    backbone_lr_mult: float = 0.1          # Backbone LR multiplier (if not using LLRD)
    
    # Hardware
    device: str = "cuda"
    num_workers: int = 8
    pin_memory: bool = True
    use_amp: bool = True
    
    # Reproducibility
    seed: int = 42
    
    def __post_init__(self):
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        
        if isinstance(self.model, dict):
            # Convert list to tuple for multiscale_stages if present
            if 'multiscale_stages' in self.model and isinstance(self.model['multiscale_stages'], list):
                self.model['multiscale_stages'] = tuple(self.model['multiscale_stages'])
            self.model = ModelConfig(**self.model)
        if isinstance(self.loss, dict):
            self.loss = LossConfig(**self.loss)
        
        # Fallback to original paths if resized versions don't exist
        if not Path(self.wildlife_root).exists():
            original = self.wildlife_root.replace('_resized', '')
            if Path(original).exists():
                print(f"Note: Using original images at {original} (resized not found)")
                self.wildlife_root = original
        
        if not Path(self.star_dataset_root).exists():
            original = self.star_dataset_root.replace('_resized', '')
            if Path(original).exists():
                print(f"Note: Using original images at {original} (resized not found)")
                self.star_dataset_root = original
    
    def save(self, path: Path):
        """Save config to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = {
            'wildlife_root': self.wildlife_root,
            'star_dataset_root': self.star_dataset_root,
            'checkpoint_dir': self.checkpoint_dir,
            'log_dir': self.log_dir,
            'model': {
                'backbone': self.model.backbone,
                'name': self.model.name,
                'use_multiscale': self.model.use_multiscale,
                'multiscale_stages': list(self.model.multiscale_stages),
                'embedding_dim': self.model.embedding_dim,
                'embedding_head_depth': self.model.embedding_head_depth,
                'use_residual_head': self.model.use_residual_head,
                'dropout': self.model.dropout,
                'use_attention_pooling': self.model.use_attention_pooling,
                'use_bnneck': self.model.use_bnneck,
                'image_size': self.model.image_size,
                'pretrained': self.model.pretrained,
                'use_tta': self.model.use_tta,
                'use_reranking': self.model.use_reranking,
            },
            'loss': {
                'circle_weight': self.loss.circle_weight,
                'triplet_weight': self.loss.triplet_weight,
                'circle_margin': self.loss.circle_margin,
                'circle_scale': self.loss.circle_scale,
                'triplet_margin': self.loss.triplet_margin,
            },
            'batch_size': self.batch_size,
            'num_instances': self.num_instances,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'warmup_epochs': self.warmup_epochs,
            'warmup_ratio': self.warmup_ratio,
            'grad_clip_norm': self.grad_clip_norm,
            'use_llrd': self.use_llrd,
            'llrd_decay': self.llrd_decay,
            'backbone_lr_mult': self.backbone_lr_mult,
            'device': self.device,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
            'use_amp': self.use_amp,
            'seed': self.seed,
        }
        
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> 'MegaStarConfig':
        """Load config from JSON."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


@dataclass
class PretrainConfig(MegaStarConfig):
    """Configuration for pre-training on Wildlife10k."""
    
    # Pre-training specific
    num_epochs: int = 50
    val_every_n_epochs: int = 5
    
    # Which wildlife datasets to use (empty = all except sea stars)
    include_datasets: List[str] = field(default_factory=list)
    exclude_datasets: List[str] = field(default_factory=lambda: ["SeaStarReID2023"])
    
    # Split strategy for wildlife data
    split_strategy: str = "original"  # original, recommended, random
    
    def __post_init__(self):
        super().__post_init__()
        self.checkpoint_dir = str(Path(self.checkpoint_dir) / "pretrain")
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)


@dataclass
class FinetuneConfig(MegaStarConfig):
    """Configuration for fine-tuning on star_dataset."""
    
    # Fine-tuning specific
    num_epochs: int = 100
    val_every_n_epochs: int = 5
    
    # Checkpoint to load (from pre-training)
    pretrain_checkpoint: Optional[str] = None
    
    # Learning rate schedule for fine-tuning (often lower than pre-training)
    learning_rate: float = 5e-5
    warmup_epochs: int = 3
    
    # Whether to freeze backbone initially
    freeze_backbone_epochs: int = 0  # 0 = no freezing
    
    # Temporal splitting for star_dataset
    train_outing_ratio: float = 0.8
    min_outings_for_eval: int = 2
    
    # Whether to include negative-only identities (single-outing stars) in training
    # These can provide positive pairs but cannot be evaluated for cross-time re-ID
    include_negative_only: bool = False
    
    def __post_init__(self):
        super().__post_init__()
        self.checkpoint_dir = str(Path(self.checkpoint_dir).parent / "finetune")
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)


@dataclass
class CotrainConfig(MegaStarConfig):
    """Configuration for co-training on Wildlife10k + star_dataset."""
    
    # Co-training specific
    num_epochs: int = 100
    val_every_n_epochs: int = 5
    
    # Batch composition
    # star_batch_ratio: fraction of each batch from star_dataset
    star_batch_ratio: float = 0.3  # 30% stars, 70% wildlife
    
    # Alternatively: alternate batches
    # If True, alternates full batches between datasets
    # If False, mixes samples within each batch
    alternate_batches: bool = False
    
    # Wildlife dataset settings
    include_datasets: List[str] = field(default_factory=list)
    exclude_datasets: List[str] = field(default_factory=lambda: ["SeaStarReID2023"])
    split_strategy: str = "original"
    
    # Star dataset temporal split
    train_outing_ratio: float = 0.8
    min_outings_for_eval: int = 2
    
    # Whether to include negative-only identities (single-outing stars) in training
    include_negative_only: bool = False
    
    # Loss weighting per dataset (optional curriculum)
    wildlife_loss_weight: float = 1.0
    star_loss_weight: float = 1.0
    
    def __post_init__(self):
        super().__post_init__()
        self.checkpoint_dir = str(Path(self.checkpoint_dir).parent / "cotrain")
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)

