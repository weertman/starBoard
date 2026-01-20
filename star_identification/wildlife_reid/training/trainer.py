"""
Training loop for Wildlife ReID.

Similar to temporal_reid trainer but simplified (no negative-only handling).
"""
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.amp import GradScaler, autocast
from tqdm import tqdm

warnings.filterwarnings('ignore', message='.*epoch parameter.*scheduler.step.*')

from .losses import CombinedLoss
from ..utils.metrics import compute_reid_metrics


class Trainer:
    """
    Trainer for Wildlife ReID model.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config,
        device: torch.device,
        gpu_ids: Optional[List[int]] = None,
    ):
        self.config = config
        self.device = device
        self.gpu_ids = gpu_ids
        
        # Multi-GPU support
        if gpu_ids and len(gpu_ids) > 1:
            print(f"Using DataParallel on GPUs: {gpu_ids}")
            model = model.to(device)
            self.model = nn.DataParallel(model, device_ids=gpu_ids)
            self.is_parallel = True
        else:
            self.model = model.to(device)
            self.is_parallel = False
        
        # Loss function
        self.loss_fn = CombinedLoss(
            circle_weight=config.circle_weight,
            triplet_weight=config.triplet_weight,
        ).to(device)
        
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # LR scheduler: warmup + cosine
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=config.warmup_epochs,
        )
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs - config.warmup_epochs,
            eta_min=1e-6,
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[config.warmup_epochs],
        )
        
        # Mixed precision
        self.use_amp = config.use_amp
        self.scaler = GradScaler('cuda') if self.use_amp else None
        
        # Tracking
        self.best_metric = 0.0
        self.best_epoch = 0
        
        # Checkpoint directory
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def train_epoch(self, train_loader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_losses = {}
        num_batches = 0
        
        pbar = tqdm(train_loader, desc="Training", leave=False)
        
        for batch in pbar:
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with autocast('cuda'):
                    embeddings = self.model(images, return_normalized=False)
                    loss, loss_dict = self.loss_fn(embeddings, labels)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                embeddings = self.model(images, return_normalized=False)
                loss, loss_dict = self.loss_fn(embeddings, labels)
                
                loss.backward()
                self.optimizer.step()
            
            # Accumulate losses
            for k, v in loss_dict.items():
                total_losses[k] = total_losses.get(k, 0.0) + v
            num_batches += 1
            
            pbar.set_postfix({
                'loss': f"{loss_dict['total']:.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}",
            })
        
        avg_losses = {k: v / num_batches for k, v in total_losses.items()}
        return avg_losses
    
    def validate(self, test_loader) -> Dict[str, float]:
        """Run validation."""
        return compute_reid_metrics(
            self.model,
            test_loader,
            self.device,
        )
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save checkpoint."""
        model_to_save = self.model.module if self.is_parallel else self.model
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
        }
        
        # Save latest
        latest_path = self.checkpoint_dir / 'latest.pth'
        torch.save(checkpoint, latest_path)
        
        # Save best
        if is_best:
            best_path = self.checkpoint_dir / 'best.pth'
            torch.save(checkpoint, best_path)
            print(f"  Saved best model (mAP: {metrics['mAP']:.4f})")
    
    def load_checkpoint(self, path: str):
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        model_to_load = self.model.module if self.is_parallel else self.model
        model_to_load.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint['epoch'], checkpoint.get('metrics', {})
    
    def train(
        self,
        train_loader,
        test_loader,
        num_epochs: Optional[int] = None,
        val_every_n_epochs: int = 5,
    ) -> Dict[str, float]:
        """Full training loop."""
        num_epochs = num_epochs or self.config.num_epochs
        
        print(f"\nStarting training for {num_epochs} epochs")
        print(f"  Validation every {val_every_n_epochs} epochs")
        print(f"  Checkpoints: {self.checkpoint_dir}")
        
        best_metrics = {}
        
        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            
            train_losses = self.train_epoch(train_loader)
            self.scheduler.step()
            
            epoch_time = time.time() - epoch_start
            loss_str = " | ".join(f"{k}: {v:.4f}" for k, v in train_losses.items())
            print(f"Epoch {epoch}/{num_epochs} ({epoch_time:.1f}s) | {loss_str}")
            
            # Validation
            if epoch % val_every_n_epochs == 0 or epoch == num_epochs:
                print("  Validating...")
                metrics = self.validate(test_loader)
                
                # Print metrics
                metric_str = " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
                print(f"  {metric_str}")
                
                is_best = metrics['mAP'] > self.best_metric
                if is_best:
                    self.best_metric = metrics['mAP']
                    self.best_epoch = epoch
                    best_metrics = metrics.copy()
                
                self.save_checkpoint(epoch, metrics, is_best=is_best)
        
        print(f"\nTraining complete!")
        print(f"Best mAP: {self.best_metric:.4f} (epoch {self.best_epoch})")
        
        return best_metrics


