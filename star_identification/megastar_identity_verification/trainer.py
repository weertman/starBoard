"""
Training loop for verification model.
"""
import json
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, List

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score

from .model import VerificationModel
from .config import TrainingConfig


class VerificationTrainer:
    """
    Trainer for verification model.
    
    Handles training loop, validation, checkpointing, and metrics.
    """
    
    def __init__(
        self,
        model: VerificationModel,
        config: TrainingConfig,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Loss function
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Optimizer
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # Scheduler: warmup + cosine
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=config.warmup_epochs,
        )
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, config.num_epochs - config.warmup_epochs),
            eta_min=config.min_lr,
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[config.warmup_epochs],
        )
        
        # AMP
        self.use_amp = config.use_amp
        self.scaler = GradScaler('cuda') if self.use_amp else None
        
        # Tracking
        self.best_metric = 0.0
        self.best_epoch = 0
        self.history: List[Dict[str, Any]] = []
        
        # Checkpoint dir
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Print model info
        params = model.get_num_params()
        print(f"Verification Model:")
        print(f"  Backbone: {params['backbone'] / 1e6:.1f}M params")
        print(f"  Cross-attention: {params['cross_attention'] / 1e6:.1f}M params")
        print(f"  Classifier: {params['classifier'] / 1e6:.2f}M params")
        print(f"  Total: {params['total'] / 1e6:.1f}M ({params['trainable'] / 1e6:.1f}M trainable)")
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(train_loader, desc="Training", leave=False)
        
        for batch in pbar:
            img_a = batch['image_a'].to(self.device)
            img_b = batch['image_b'].to(self.device)
            labels = batch['label'].to(self.device).unsqueeze(1)
            
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with autocast('cuda'):
                    logits = self.model(img_a, img_b)
                    loss = self.criterion(logits, labels)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(img_a, img_b)
                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            
            # Collect predictions
            with torch.no_grad():
                probs = torch.sigmoid(logits).cpu().numpy()
                all_preds.extend(probs.flatten())
                all_labels.extend(labels.cpu().numpy().flatten())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Compute metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(all_labels, all_preds > 0.5)
        auc = roc_auc_score(all_labels, all_preds)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'auc': auc,
        }
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Run validation."""
        self.model.eval()
        
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        for batch in tqdm(val_loader, desc="Validating", leave=False):
            img_a = batch['image_a'].to(self.device)
            img_b = batch['image_b'].to(self.device)
            labels = batch['label'].to(self.device).unsqueeze(1)
            
            if self.use_amp:
                with autocast('cuda'):
                    logits = self.model(img_a, img_b)
                    loss = self.criterion(logits, labels)
            else:
                logits = self.model(img_a, img_b)
                loss = self.criterion(logits, labels)
            
            total_loss += loss.item()
            
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(probs.flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Compute metrics at various thresholds
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_preds > 0.5)
        auc = roc_auc_score(all_labels, all_preds)
        
        # Precision/recall at 0.5 threshold
        preds_binary = (all_preds > 0.5).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, preds_binary, average='binary', zero_division=0
        )
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'auc': auc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }
    
    def save_checkpoint(
        self,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False,
    ):
        """Save checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': {
                'learning_rate': self.config.learning_rate,
                'batch_size': self.config.batch_size,
            },
        }
        
        # Save latest
        torch.save(checkpoint, self.checkpoint_dir / 'latest.pth')
        
        # Save best
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / 'best.pth')
            print(f"  [SAVED] Best model (AUC: {metrics.get('auc', 0):.4f})")
    
    def load_checkpoint(self, path: str) -> int:
        """Load checkpoint, return epoch number."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_metric = checkpoint.get('metrics', {}).get('auc', 0.0)
        return checkpoint['epoch']
    
    def unfreeze_backbone(self, new_lr: Optional[float] = None):
        """
        Unfreeze backbone parameters and reinitialize optimizer.
        
        Args:
            new_lr: New learning rate after unfreezing (default: current LR * 0.1)
        """
        # Unfreeze backbone
        for param in self.model.backbone.parameters():
            param.requires_grad = True
        
        # Get current LR
        current_lr = self.optimizer.param_groups[0]['lr']
        backbone_lr = new_lr if new_lr is not None else current_lr * 0.1
        
        # Reinitialize optimizer with all parameters
        # Use lower LR for backbone (discriminative learning rates)
        backbone_params = list(self.model.backbone.parameters())
        # Use id() for identity comparison - using `in` with tensors causes broadcast errors
        backbone_param_ids = {id(p) for p in backbone_params}
        other_params = [p for p in self.model.parameters() if p.requires_grad and id(p) not in backbone_param_ids]
        
        self.optimizer = AdamW([
            {'params': backbone_params, 'lr': backbone_lr},
            {'params': other_params, 'lr': current_lr},
        ], weight_decay=self.config.weight_decay)
        
        # Update param count
        params = self.model.get_num_params()
        print(f"\n[UNFROZEN] Backbone unfrozen!")
        print(f"   Backbone LR: {backbone_lr:.2e}, Head LR: {current_lr:.2e}")
        print(f"   Trainable params: {params['trainable'] / 1e6:.1f}M")
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: Optional[int] = None,
        val_every: int = 5,
        unfreeze_backbone_at_epoch: Optional[int] = None,
        unfreeze_backbone_lr: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs (uses config if None)
            val_every: Validate every N epochs
            unfreeze_backbone_at_epoch: Epoch at which to unfreeze backbone (None = never)
            unfreeze_backbone_lr: Learning rate for backbone after unfreezing
            
        Returns:
            Best validation metrics
        """
        num_epochs = num_epochs or self.config.num_epochs
        
        print(f"\n{'='*60}")
        print(f"VERIFICATION TRAINING")
        print(f"{'='*60}")
        print(f"Epochs: {num_epochs}, Val every: {val_every}")
        print(f"LR: {self.config.learning_rate:.2e}, Warmup: {self.config.warmup_epochs}")
        if unfreeze_backbone_at_epoch is not None:
            print(f"Unfreeze backbone at epoch: {unfreeze_backbone_at_epoch}")
        print(f"Checkpoints: {self.checkpoint_dir}")
        
        best_metrics = {}
        
        # Epoch 0: Validate before any training (baseline performance)
        print("\nEpoch 0 (baseline - no training yet)")
        print("  Validating...")
        val_metrics = self.validate(val_loader)
        print(f"  Val: loss={val_metrics['loss']:.4f} | "
              f"acc={val_metrics['accuracy']:.4f} | "
              f"auc={val_metrics['auc']:.4f} | "
              f"f1={val_metrics['f1']:.4f}")
        
        # Track as initial best
        self.best_metric = val_metrics['auc']
        self.best_epoch = 0
        best_metrics = val_metrics.copy()
        
        # Log epoch 0
        self.history.append({
            'epoch': 0,
            'epoch_time': 0,
            'lr': self.optimizer.param_groups[0]['lr'],
            **{f'val_{k}': v for k, v in val_metrics.items()},
        })
        
        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            
            # Check if we should unfreeze backbone at this epoch
            if unfreeze_backbone_at_epoch is not None and epoch == unfreeze_backbone_at_epoch:
                self.unfreeze_backbone(new_lr=unfreeze_backbone_lr)
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            self.scheduler.step()
            
            # Resample pairs for next epoch
            if hasattr(train_loader.dataset, 'on_epoch_end'):
                train_loader.dataset.on_epoch_end()
            
            epoch_time = time.time() - epoch_start
            
            print(f"Epoch {epoch}/{num_epochs} ({epoch_time:.1f}s) | "
                  f"loss: {train_metrics['loss']:.4f} | "
                  f"acc: {train_metrics['accuracy']:.4f} | "
                  f"auc: {train_metrics['auc']:.4f}")
            
            # Validate
            val_metrics = None
            if epoch % val_every == 0 or epoch == num_epochs:
                # Resample validation pairs
                if hasattr(val_loader.dataset, 'on_epoch_end'):
                    val_loader.dataset.on_epoch_end()
                
                print("  Validating...")
                val_metrics = self.validate(val_loader)
                print(f"  Val: loss={val_metrics['loss']:.4f} | "
                      f"acc={val_metrics['accuracy']:.4f} | "
                      f"auc={val_metrics['auc']:.4f} | "
                      f"f1={val_metrics['f1']:.4f}")
                
                is_best = val_metrics['auc'] > self.best_metric
                if is_best:
                    self.best_metric = val_metrics['auc']
                    self.best_epoch = epoch
                    best_metrics = val_metrics.copy()
                
                self.save_checkpoint(epoch, val_metrics, is_best)
            
            # Log history
            entry = {
                'epoch': epoch,
                'epoch_time': round(epoch_time, 2),
                'lr': self.optimizer.param_groups[0]['lr'],
                **{f'train_{k}': v for k, v in train_metrics.items()},
            }
            if val_metrics:
                entry.update({f'val_{k}': v for k, v in val_metrics.items()})
            self.history.append(entry)
            
            # Save history
            with open(self.checkpoint_dir / 'history.json', 'w') as f:
                json.dump(self.history, f, indent=2)
        
        print(f"\n[DONE] Training complete! Best AUC: {self.best_metric:.4f} (epoch {self.best_epoch})")
        
        return best_metrics

