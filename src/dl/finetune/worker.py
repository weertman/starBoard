"""
Fine-tuning worker for background training.

This module provides a QThread-based worker that runs model fine-tuning
in the background with progress reporting.
"""

from __future__ import annotations

import json
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Optional, Dict, Any

from PySide6.QtCore import QThread, Signal

from .config import FinetuneUIConfig, FinetuneMode, DataSource
from .data_bridge import (
    create_archive_metadata_csv,
    merge_with_star_dataset,
    create_verification_pairs,
    collect_archive_identities,
    get_data_summary,
)

log = logging.getLogger("starBoard.dl.finetune.worker")


# Add star_identification to path for imports
_STAR_ID_PATH = Path(__file__).parent.parent.parent.parent / "star_identification"
if str(_STAR_ID_PATH) not in sys.path:
    sys.path.insert(0, str(_STAR_ID_PATH))


class FinetuneWorker(QThread):
    """
    Background worker for fine-tuning models.
    
    Signals:
        progress(str, int, int): (message, current_epoch, total_epochs)
        finished(bool, str, str): (success, message, output_path)
    """
    
    progress = Signal(str, int, int)  # message, current, total
    finished = Signal(bool, str, str)  # success, message, output_path
    
    def __init__(self, config: FinetuneUIConfig, parent=None):
        super().__init__(parent)
        self.config = config
        self._cancelled = False
        self._start_time: Optional[float] = None
    
    def cancel(self):
        """Request cancellation of training."""
        self._cancelled = True
        log.info("Fine-tuning cancellation requested")
    
    def run(self):
        """Main worker thread."""
        try:
            self._start_time = time.time()
            self._run_training()
        except Exception as e:
            log.error("Fine-tuning failed: %s", e)
            traceback.print_exc()
            self.finished.emit(False, f"Error: {str(e)}", "")
    
    def _emit_progress(self, message: str, current: int, total: int):
        """Emit progress with timing info."""
        if self._start_time:
            elapsed = time.time() - self._start_time
            if current > 0 and total > 0:
                eta = (elapsed / current) * (total - current)
                message = f"{message} | ETA: {self._format_time(eta)}"
        self.progress.emit(message, current, total)
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds as human-readable time."""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            return f"{int(seconds / 60)}m {int(seconds % 60)}s"
        else:
            hours = int(seconds / 3600)
            minutes = int((seconds % 3600) / 60)
            return f"{hours}h {minutes}m"
    
    def _run_training(self):
        """Run the training based on mode."""
        # Validate configuration
        valid, error = self.config.validate()
        if not valid:
            self.finished.emit(False, error, "")
            return
        
        self._emit_progress("Initializing...", 0, self.config.epochs)
        
        # Check data availability
        data_summary = get_data_summary()
        if not data_summary["cache_exists"]:
            self.finished.emit(
                False,
                "Image cache not found. Run precomputation first.",
                ""
            )
            return
        
        if data_summary["total_images"] < 10:
            self.finished.emit(
                False,
                f"Not enough images for training ({data_summary['total_images']} found). Need at least 10.",
                ""
            )
            return
        
        if self._cancelled:
            self.finished.emit(False, "Cancelled", "")
            return
        
        # Route to appropriate training method
        if self.config.mode == FinetuneMode.EMBEDDING:
            self._run_embedding_training()
        else:
            self._run_verification_training()
    
    def _run_embedding_training(self):
        """Run embedding model fine-tuning."""
        try:
            import torch
            from megastarid.config import FinetuneConfig, ModelConfig, LossConfig
            from megastarid.datasets import create_finetune_dataloaders
            from megastarid.models import create_model, load_pretrained_model, freeze_backbone, unfreeze_backbone
            from megastarid.trainer import MegaStarTrainer
        except ImportError as e:
            self.finished.emit(False, f"Failed to import training modules: {e}", "")
            return
        
        self._emit_progress("Preparing data...", 0, self.config.epochs)
        
        # Create output directory
        output_dir = self.config.get_output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare data directory with metadata
        data_dir = output_dir / "data"
        
        # Create archive metadata
        identities = collect_archive_identities()
        metadata_path = create_archive_metadata_csv(data_dir, identities)
        
        # Optionally merge with star_dataset
        if self.config.data_source == DataSource.ARCHIVE_PLUS_STAR and self.config.star_dataset_path:
            star_path = Path(self.config.star_dataset_path)
            if star_path.exists():
                metadata_path = merge_with_star_dataset(
                    metadata_path, star_path, data_dir
                )
        
        if self._cancelled:
            self.finished.emit(False, "Cancelled", "")
            return
        
        self._emit_progress("Loading model...", 0, self.config.epochs)
        
        # Device setup
        device = torch.device(
            self.config.device if torch.cuda.is_available() else "cpu"
        )
        
        # Get base model checkpoint path from registry
        from src.dl.registry import DLRegistry
        registry = DLRegistry.load()
        
        base_entry = registry.models.get(self.config.base_model_key)
        if not base_entry:
            self.finished.emit(False, f"Base model not found: {self.config.base_model_key}", "")
            return
        
        # Create training config
        train_config = FinetuneConfig(
            star_dataset_root=str(data_dir),
            checkpoint_dir=str(output_dir),
            pretrain_checkpoint=base_entry.checkpoint_path,
            model=ModelConfig(
                backbone=base_entry.backbone or "convnext-tiny",
                embedding_dim=base_entry.embedding_dim or 512,
                image_size=384,
            ),
            loss=LossConfig(
                circle_weight=1.0,
                triplet_weight=0.0,
            ),
            num_epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            device=self.config.device,
            num_workers=self.config.num_workers,
            use_amp=self.config.use_amp,
            freeze_backbone_epochs=self.config.freeze_backbone_epochs,
        )
        
        if self._cancelled:
            self.finished.emit(False, "Cancelled", "")
            return
        
        # Create dataloaders
        self._emit_progress("Creating dataloaders...", 0, self.config.epochs)
        
        try:
            train_loader, gallery_loader, query_loader = create_finetune_dataloaders(train_config)
        except Exception as e:
            log.error("Failed to create dataloaders: %s", e)
            self.finished.emit(False, f"Failed to create dataloaders: {e}", "")
            return
        
        if self._cancelled:
            self.finished.emit(False, "Cancelled", "")
            return
        
        # Load model
        self._emit_progress("Loading pretrained model...", 0, self.config.epochs)
        
        try:
            model = load_pretrained_model(train_config, base_entry.checkpoint_path, device)
        except Exception as e:
            log.error("Failed to load model: %s", e)
            self.finished.emit(False, f"Failed to load model: {e}", "")
            return
        
        if self._cancelled:
            self.finished.emit(False, "Cancelled", "")
            return
        
        # Create trainer
        trainer = MegaStarTrainer(
            model=model,
            config=train_config,
            device=device,
            mode='finetune',
        )
        
        # Optional backbone freezing
        if self.config.freeze_backbone_epochs > 0:
            freeze_backbone(model)
        
        # Training loop with progress reporting
        best_metrics: Dict[str, float] = {}
        
        for epoch in range(1, self.config.epochs + 1):
            if self._cancelled:
                self.finished.emit(False, "Cancelled", "")
                return
            
            self._emit_progress(f"Training epoch {epoch}...", epoch, self.config.epochs)
            
            # Unfreeze backbone after frozen epochs
            if epoch == self.config.freeze_backbone_epochs + 1:
                unfreeze_backbone(model)
                # Reset optimizer with lower LR
                trainer.optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=self.config.learning_rate / 10,
                    weight_decay=train_config.weight_decay,
                )
            
            # Train one epoch
            try:
                train_losses = trainer.train_epoch(train_loader)
                trainer.scheduler.step()
            except Exception as e:
                log.error("Training epoch failed: %s", e)
                self.finished.emit(False, f"Training failed at epoch {epoch}: {e}", "")
                return
            
            # Validate periodically
            if epoch % 5 == 0 or epoch == self.config.epochs:
                try:
                    val_metrics = trainer.validate_star(gallery_loader, query_loader)
                    
                    is_best = val_metrics.get('mAP', 0) > best_metrics.get('mAP', 0)
                    if is_best:
                        best_metrics = val_metrics.copy()
                        # Save best checkpoint
                        trainer.save_checkpoint(epoch, val_metrics, is_best=True)
                except Exception as e:
                    log.warning("Validation failed: %s", e)
        
        # Save final checkpoint
        trainer.save_checkpoint(self.config.epochs, best_metrics, is_best=False)
        
        # Save config
        train_config.save(output_dir / "config.json")
        
        # Save training summary
        summary = {
            "mode": "embedding",
            "base_model": self.config.base_model_key,
            "epochs": self.config.epochs,
            "best_metrics": best_metrics,
            "output_dir": str(output_dir),
        }
        with open(output_dir / "training_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        self.finished.emit(
            True,
            f"Training complete! Best mAP: {best_metrics.get('mAP', 0):.4f}",
            str(output_dir / "best.pth")
        )
    
    def _run_verification_training(self):
        """Run verification model fine-tuning."""
        try:
            import torch
            from megastar_identity_verification.config import (
                VerificationConfig, BackboneConfig, CrossAttentionConfig, FinetuneConfig
            )
            from megastar_identity_verification.model import VerificationModel, create_verification_model
            from megastar_identity_verification.trainer import VerificationTrainer
            from megastar_identity_verification.transforms import get_train_transforms, get_test_transforms
        except ImportError as e:
            self.finished.emit(False, f"Failed to import verification modules: {e}", "")
            return
        
        self._emit_progress("Preparing verification data...", 0, self.config.epochs)
        
        # Create output directory
        output_dir = self.config.get_output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create verification pairs
        data_dir = output_dir / "data"
        train_pairs_path, val_pairs_path = create_verification_pairs(
            data_dir,
            positive_ratio=0.5,
            pairs_per_split=self.config.pairs_per_epoch,
        )
        
        if self._cancelled:
            self.finished.emit(False, "Cancelled", "")
            return
        
        self._emit_progress("Loading verification model...", 0, self.config.epochs)
        
        # Device setup
        device = torch.device(
            self.config.device if torch.cuda.is_available() else "cpu"
        )
        
        # Get base model checkpoint path from registry
        from src.dl.registry import DLRegistry
        registry = DLRegistry.load()
        
        base_entry = registry.verification_models.get(self.config.base_model_key)
        if not base_entry:
            # Try embedding models if not found in verification
            emb_entry = registry.models.get(self.config.base_model_key)
            if emb_entry:
                # Use embedding model as backbone for new verification model
                backbone_checkpoint = emb_entry.checkpoint_path
                use_backbone = True
            else:
                self.finished.emit(False, f"Base model not found: {self.config.base_model_key}", "")
                return
        else:
            backbone_checkpoint = base_entry.checkpoint_path
            use_backbone = False
        
        # Model config
        model_config = VerificationConfig(
            backbone=BackboneConfig(
                name="convnext-small",
                freeze=self.config.freeze_backbone,
            ),
            cross_attention=CrossAttentionConfig(
                feature_dim=768,
                hidden_dim=256,
                num_layers=2,
                num_heads=8,
            ),
        )
        
        if self._cancelled:
            self.finished.emit(False, "Cancelled", "")
            return
        
        # Create or load model
        try:
            if use_backbone:
                # Create fresh verification model with embedding backbone
                model = create_verification_model(
                    config=model_config,
                    backbone_checkpoint=backbone_checkpoint,
                )
            else:
                # Load full verification model
                checkpoint = torch.load(backbone_checkpoint, map_location='cpu', weights_only=False)
                model = VerificationModel(model_config)
                model.load_state_dict(checkpoint['model_state_dict'])
        except Exception as e:
            log.error("Failed to load/create model: %s", e)
            self.finished.emit(False, f"Failed to load model: {e}", "")
            return
        
        if self._cancelled:
            self.finished.emit(False, "Cancelled", "")
            return
        
        # Create datasets
        self._emit_progress("Creating verification datasets...", 0, self.config.epochs)
        
        try:
            train_transform = get_train_transforms(224)
            test_transform = get_test_transforms(224)
            
            # Use our CSV-based pair dataset
            train_dataset = PairDatasetFromCSV(
                csv_path=str(train_pairs_path),
                transform=train_transform,
            )
            val_dataset = PairDatasetFromCSV(
                csv_path=str(val_pairs_path),
                transform=test_transform,
            )
            
            from torch.utils.data import DataLoader
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=self.config.num_workers,
                pin_memory=True,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                pin_memory=True,
            )
        except Exception as e:
            log.error("Failed to create datasets: %s", e)
            self.finished.emit(False, f"Failed to create datasets: {e}", "")
            return
        
        if self._cancelled:
            self.finished.emit(False, "Cancelled", "")
            return
        
        # Training config
        train_config = FinetuneConfig(
            checkpoint_dir=str(output_dir),
            batch_size=self.config.batch_size,
            num_epochs=self.config.epochs,
            learning_rate=self.config.learning_rate,
            num_workers=self.config.num_workers,
            use_amp=self.config.use_amp,
        )
        
        # Create trainer
        trainer = VerificationTrainer(model, train_config, device)
        
        # Training loop
        best_metrics: Dict[str, float] = {}
        
        for epoch in range(1, self.config.epochs + 1):
            if self._cancelled:
                self.finished.emit(False, "Cancelled", "")
                return
            
            self._emit_progress(f"Training epoch {epoch}...", epoch, self.config.epochs)
            
            # Train one epoch
            try:
                train_metrics = trainer.train_epoch(train_loader)
            except Exception as e:
                log.error("Training epoch failed: %s", e)
                self.finished.emit(False, f"Training failed at epoch {epoch}: {e}", "")
                return
            
            trainer.scheduler.step()
            
            # Validate periodically
            if epoch % 5 == 0 or epoch == self.config.epochs:
                try:
                    val_metrics = trainer.validate(val_loader)
                    
                    is_best = val_metrics.get('auc', 0) > best_metrics.get('auc', 0)
                    if is_best:
                        best_metrics = val_metrics.copy()
                        # Save best checkpoint
                        trainer.save_checkpoint(epoch, val_metrics, is_best=True)
                except Exception as e:
                    log.warning("Validation failed: %s", e)
        
        # Save final checkpoint
        trainer.save_checkpoint(self.config.epochs, best_metrics, is_best=False)
        
        # Save training summary
        summary = {
            "mode": "verification",
            "base_model": self.config.base_model_key,
            "epochs": self.config.epochs,
            "best_metrics": best_metrics,
            "output_dir": str(output_dir),
        }
        with open(output_dir / "training_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        self.finished.emit(
            True,
            f"Training complete! Best AUC: {best_metrics.get('auc', 0):.4f}",
            str(output_dir / "best.pth")
        )


class PairDatasetFromCSV:
    """
    Simple pair dataset that reads from a CSV file.
    
    Implements __len__ and __getitem__ for torch DataLoader compatibility.
    """
    
    def __init__(self, csv_path: str, transform=None):
        import csv
        
        self.transform = transform
        self.pairs = []
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.pairs.append({
                    'image_a': row['image_a'],
                    'image_b': row['image_b'],
                    'label': int(row['label']),
                })
        
        log.info("PairDatasetFromCSV loaded %d pairs from %s", len(self.pairs), csv_path)
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        import torch
        from PIL import Image
        
        pair = self.pairs[idx]
        
        try:
            img_a = Image.open(pair['image_a']).convert('RGB')
            img_b = Image.open(pair['image_b']).convert('RGB')
        except Exception as e:
            log.warning("Failed to load pair %d: %s", idx, e)
            # Return placeholder images
            img_a = Image.new('RGB', (224, 224), (128, 128, 128))
            img_b = Image.new('RGB', (224, 224), (128, 128, 128))
        
        if self.transform:
            img_a = self.transform(img_a)
            img_b = self.transform(img_b)
        
        return {
            'image_a': img_a,
            'image_b': img_b,
            'label': torch.tensor(pair['label'], dtype=torch.float32),
        }

