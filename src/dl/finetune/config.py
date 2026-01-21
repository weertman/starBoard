"""
Fine-tuning configuration for UI integration.

This module provides configuration dataclasses that bridge the UI controls
to the underlying training infrastructure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional
from datetime import datetime


class FinetuneMode(Enum):
    """Training mode selection."""
    EMBEDDING = "embedding"      # Circle Loss embedding model (first-order ranking)
    VERIFICATION = "verification"  # Pairwise verification model


class DataSource(Enum):
    """Data source selection."""
    ARCHIVE_ONLY = "archive_only"          # Use only starBoard archive data
    ARCHIVE_PLUS_STAR = "archive_plus_star"  # Merge archive with star_dataset


@dataclass
class FinetuneUIConfig:
    """
    Configuration for fine-tuning from the UI.
    
    This is a simplified configuration that captures user selections
    from the UI and gets translated to the appropriate training config.
    """
    
    # Mode: what type of model to fine-tune
    mode: FinetuneMode = FinetuneMode.EMBEDDING
    
    # Base model to fine-tune from
    base_model_key: str = ""
    
    # Output name for the fine-tuned model
    output_name: str = ""
    
    # Data source configuration
    data_source: DataSource = DataSource.ARCHIVE_ONLY
    star_dataset_path: Optional[str] = None  # Path to star_dataset_resized if using
    
    # Training hyperparameters
    epochs: int = 25
    learning_rate: float = 5e-5
    batch_size: int = 8
    
    # Hardware settings
    device: str = "cuda"
    num_workers: int = 4
    use_amp: bool = True
    
    # Advanced options (embedded model)
    freeze_backbone_epochs: int = 0  # Epochs to freeze backbone
    
    # Advanced options (verification model)
    freeze_backbone: bool = False  # Freeze backbone entirely
    pairs_per_epoch: int = 10000
    
    # Output directory (auto-generated if not specified)
    output_dir: Optional[str] = None
    
    def __post_init__(self):
        """Validate and set defaults after initialization."""
        # Generate output name if not provided
        if not self.output_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            mode_str = self.mode.value
            self.output_name = f"finetuned_{mode_str}_{timestamp}"
        
        # Set default star_dataset path if using it but path not provided
        if self.data_source == DataSource.ARCHIVE_PLUS_STAR and not self.star_dataset_path:
            # Look for common locations
            default_paths = [
                Path("star_identification/star_dataset_resized"),
                Path("star_dataset_resized"),
                Path("D:/star_identification/star_dataset_resized"),
            ]
            for p in default_paths:
                if p.exists():
                    self.star_dataset_path = str(p)
                    break
    
    def get_output_dir(self) -> Path:
        """Get the output directory for checkpoints."""
        if self.output_dir:
            return Path(self.output_dir)
        
        # Generate based on mode
        base = Path("star_identification/checkpoints")
        if self.mode == FinetuneMode.EMBEDDING:
            return base / "megastarid" / self.output_name
        else:
            return base / "verification" / self.output_name
    
    def get_registry_key(self) -> str:
        """Get the key to use when registering in DLRegistry."""
        # Sanitize the output name for use as a key
        key = self.output_name.lower().replace(" ", "_").replace("-", "_")
        return key
    
    def validate(self) -> tuple[bool, str]:
        """
        Validate the configuration.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self.base_model_key:
            return False, "No base model selected"
        
        if self.epochs < 1:
            return False, "Epochs must be at least 1"
        
        if self.batch_size < 1:
            return False, "Batch size must be at least 1"
        
        if self.learning_rate <= 0:
            return False, "Learning rate must be positive"
        
        if self.data_source == DataSource.ARCHIVE_PLUS_STAR:
            if self.star_dataset_path:
                p = Path(self.star_dataset_path)
                if not p.exists():
                    return False, f"star_dataset path not found: {self.star_dataset_path}"
        
        return True, ""
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "mode": self.mode.value,
            "base_model_key": self.base_model_key,
            "output_name": self.output_name,
            "data_source": self.data_source.value,
            "star_dataset_path": self.star_dataset_path,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "device": self.device,
            "num_workers": self.num_workers,
            "use_amp": self.use_amp,
            "freeze_backbone_epochs": self.freeze_backbone_epochs,
            "freeze_backbone": self.freeze_backbone,
            "pairs_per_epoch": self.pairs_per_epoch,
            "output_dir": self.output_dir,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "FinetuneUIConfig":
        """Create from dictionary."""
        return cls(
            mode=FinetuneMode(d.get("mode", "embedding")),
            base_model_key=d.get("base_model_key", ""),
            output_name=d.get("output_name", ""),
            data_source=DataSource(d.get("data_source", "archive_only")),
            star_dataset_path=d.get("star_dataset_path"),
            epochs=d.get("epochs", 25),
            learning_rate=d.get("learning_rate", 5e-5),
            batch_size=d.get("batch_size", 8),
            device=d.get("device", "cuda"),
            num_workers=d.get("num_workers", 4),
            use_amp=d.get("use_amp", True),
            freeze_backbone_epochs=d.get("freeze_backbone_epochs", 0),
            freeze_backbone=d.get("freeze_backbone", False),
            pairs_per_epoch=d.get("pairs_per_epoch", 10000),
            output_dir=d.get("output_dir"),
        )




