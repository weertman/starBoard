"""
Unified trainer for MegaStarID.

Supports pre-training, fine-tuning, and co-training modes.

Features:
- Gradient clipping for training stability
- Layer-wise learning rate decay (LLRD) for transformers
- Architecture-specific optimizer configuration
- Warmup ratio support
"""
import json
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np

warnings.filterwarnings('ignore', message='.*epoch parameter.*scheduler.step.*')
warnings.filterwarnings('ignore', message='.*call of `lr_scheduler.step\\(\\)` before `optimizer.step\\(\\)`.*')

# Import from wildlife_reid
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from wildlife_reid.training.losses import CombinedLoss
from wildlife_reid.utils.metrics import compute_reid_metrics

from megastarid.inference import compute_reid_metrics_with_enhancements


def get_swinv2_layer_groups(model) -> List[Dict[str, Any]]:
    """
    Get parameter groups for SwinV2 with layer-wise learning rate decay.
    
    Groups from deepest (lowest LR) to shallowest (highest LR):
    - Embedding/patch layers (deepest)
    - Stage 1 layers
    - Stage 2 layers
    - Stage 3 layers
    - Stage 4 layers (shallowest backbone layer)
    - Head layers (highest LR)
    
    Args:
        model: MultiScaleReID model with SwinV2 backbone
        
    Returns:
        List of (params, layer_name) for each group
    """
    groups = []
    
    # Get backbone - handle both wrapped and unwrapped models
    backbone_module = model
    if hasattr(model, 'module'):
        backbone_module = model.module
    
    if not hasattr(backbone_module, 'backbone'):
        return []
    
    backbone = backbone_module.backbone
    
    # Check if it's SwinV2
    if not hasattr(backbone, 'backbone'):
        return []
    
    swin = backbone.backbone  # The actual Swinv2Model
    
    # Group 0: Embeddings (patch embedding, etc.) - deepest
    embed_params = []
    if hasattr(swin, 'embeddings'):
        embed_params.extend(list(swin.embeddings.parameters()))
    groups.append({'params': embed_params, 'name': 'embeddings'})
    
    # Groups 1-4: Encoder stages
    if hasattr(swin, 'encoder') and hasattr(swin.encoder, 'layers'):
        for i, layer in enumerate(swin.encoder.layers):
            layer_params = list(layer.parameters())
            groups.append({'params': layer_params, 'name': f'encoder_layer_{i}'})
    
    # Final group: All other backbone parts (layernorm, etc.)
    counted_params = set()
    for g in groups:
        for p in g['params']:
            counted_params.add(id(p))
    
    other_backbone_params = []
    for p in backbone.parameters():
        if id(p) not in counted_params:
            other_backbone_params.append(p)
    if other_backbone_params:
        groups.append({'params': other_backbone_params, 'name': 'backbone_other'})
    
    # Head layers (highest LR) - everything not in backbone
    head_params = []
    backbone_param_ids = set(id(p) for p in backbone.parameters())
    for p in backbone_module.parameters():
        if id(p) not in backbone_param_ids:
            head_params.append(p)
    groups.append({'params': head_params, 'name': 'head'})
    
    return groups


def create_optimizer_with_llrd(
    model: nn.Module,
    base_lr: float,
    weight_decay: float,
    llrd_decay: float = 0.75,
    backbone_name: str = 'swinv2-tiny',
) -> AdamW:
    """
    Create AdamW optimizer with layer-wise learning rate decay.
    
    For transformers, deeper layers get lower learning rates.
    
    Args:
        model: The model to optimize
        base_lr: Learning rate for the head (highest LR)
        weight_decay: Weight decay
        llrd_decay: Decay factor per layer (e.g., 0.75 means each layer gets 0.75x the LR of the layer above)
        backbone_name: Backbone architecture name
        
    Returns:
        AdamW optimizer with per-layer learning rates
    """
    if 'swin' in backbone_name.lower():
        layer_groups = get_swinv2_layer_groups(model)
        
        if not layer_groups:
            # Fallback to simple optimizer
            return AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=base_lr,
                weight_decay=weight_decay,
            )
        
        # Assign learning rates: head gets base_lr, each layer below gets llrd_decay * previous
        # Groups are ordered: embeddings, encoder_layer_0, ..., encoder_layer_N, backbone_other, head
        num_groups = len(layer_groups)
        param_groups = []
        
        for i, group in enumerate(layer_groups):
            # Distance from head (last group)
            depth = num_groups - 1 - i
            lr = base_lr * (llrd_decay ** depth)
            
            if group['params']:  # Only add non-empty groups
                param_groups.append({
                    'params': [p for p in group['params'] if p.requires_grad],
                    'lr': lr,
                    'weight_decay': weight_decay,
                    'name': group['name'],
                })
        
        # Filter out empty groups
        param_groups = [g for g in param_groups if g['params']]
        
        if param_groups:
            print(f"  LLRD enabled with {len(param_groups)} parameter groups:")
            for g in param_groups:
                print(f"    {g['name']}: lr={g['lr']:.2e}, params={len(g['params'])}")
            
            return AdamW(param_groups)
    
    # Fallback for non-transformer architectures
    return AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=base_lr,
        weight_decay=weight_decay,
    )


def create_optimizer_with_backbone_mult(
    model: nn.Module,
    base_lr: float,
    weight_decay: float,
    backbone_lr_mult: float = 0.1,
) -> AdamW:
    """
    Create AdamW optimizer with separate backbone/head learning rates.
    
    Simpler alternative to LLRD - just uses two LR groups.
    
    Args:
        model: The model to optimize
        base_lr: Learning rate for the head
        weight_decay: Weight decay
        backbone_lr_mult: Multiplier for backbone LR (e.g., 0.1 = backbone at 10% of head LR)
        
    Returns:
        AdamW optimizer
    """
    # Get backbone - handle both wrapped and unwrapped models
    backbone_module = model
    if hasattr(model, 'module'):
        backbone_module = model.module
    
    backbone_params = []
    head_params = []
    
    if hasattr(backbone_module, 'backbone'):
        backbone = backbone_module.backbone
        backbone_param_ids = set(id(p) for p in backbone.parameters())
        
        for p in backbone_module.parameters():
            if p.requires_grad:
                if id(p) in backbone_param_ids:
                    backbone_params.append(p)
                else:
                    head_params.append(p)
    else:
        # Fallback: all params at base_lr
        return AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=base_lr,
            weight_decay=weight_decay,
        )
    
    param_groups = []
    if backbone_params:
        param_groups.append({
            'params': backbone_params,
            'lr': base_lr * backbone_lr_mult,
            'weight_decay': weight_decay,
            'name': 'backbone',
        })
    if head_params:
        param_groups.append({
            'params': head_params,
            'lr': base_lr,
            'weight_decay': weight_decay,
            'name': 'head',
        })
    
    if param_groups:
        print(f"  Backbone/Head LR split:")
        for g in param_groups:
            print(f"    {g['name']}: lr={g['lr']:.2e}, params={len(g['params'])}")
        
        return AdamW(param_groups)
    
    # Fallback
    return AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=base_lr,
        weight_decay=weight_decay,
    )


class MegaStarTrainer:
    """
    Unified trainer for all MegaStarID training modes.
    
    Features:
    - Architecture-specific optimizer configuration
    - Layer-wise learning rate decay (LLRD) for transformers
    - Gradient clipping for stability
    - Warmup ratio support
    """
    
    def __init__(
        self,
        model: nn.Module,
        config,
        device: torch.device,
        mode: str = 'pretrain',  # 'pretrain', 'finetune', 'cotrain'
    ):
        self.config = config
        self.device = device
        self.mode = mode
        self.model = model.to(device)
        
        # Loss function
        self.loss_fn = CombinedLoss(
            circle_weight=config.loss.circle_weight,
            triplet_weight=config.loss.triplet_weight,
            circle_margin=config.loss.circle_margin,
            circle_scale=config.loss.circle_scale,
            triplet_margin=config.loss.triplet_margin,
        ).to(device)
        
        # Gradient clipping
        self.grad_clip_norm = getattr(config, 'grad_clip_norm', None)
        if self.grad_clip_norm:
            print(f"  Gradient clipping enabled: max_norm={self.grad_clip_norm}")
        
        # Optimizer with optional LLRD or backbone LR multiplier
        use_llrd = getattr(config, 'use_llrd', False)
        backbone_name = getattr(config.model, 'backbone', 'swinv2-tiny')
        
        if use_llrd and 'swin' in backbone_name.lower():
            llrd_decay = getattr(config, 'llrd_decay', 0.75)
            print(f"  Using layer-wise LR decay (LLRD) with decay={llrd_decay}")
            self.optimizer = create_optimizer_with_llrd(
                self.model,
                base_lr=config.learning_rate,
                weight_decay=config.weight_decay,
                llrd_decay=llrd_decay,
                backbone_name=backbone_name,
            )
        else:
            backbone_lr_mult = getattr(config, 'backbone_lr_mult', 1.0)
            if backbone_lr_mult < 1.0:
                print(f"  Using backbone LR multiplier: {backbone_lr_mult}")
                self.optimizer = create_optimizer_with_backbone_mult(
                    self.model,
                    base_lr=config.learning_rate,
                    weight_decay=config.weight_decay,
                    backbone_lr_mult=backbone_lr_mult,
                )
            else:
                # Standard optimizer
                self.optimizer = AdamW(
                    filter(lambda p: p.requires_grad, self.model.parameters()),
                    lr=config.learning_rate,
                    weight_decay=config.weight_decay,
                )
        
        # Compute warmup epochs from ratio if specified
        warmup_ratio = getattr(config, 'warmup_ratio', None)
        if warmup_ratio is not None and warmup_ratio > 0:
            warmup_epochs = max(1, int(config.num_epochs * warmup_ratio))
            print(f"  Warmup: {warmup_epochs} epochs ({warmup_ratio:.0%} of {config.num_epochs})")
        else:
            warmup_epochs = getattr(config, 'warmup_epochs', 5)
            print(f"  Warmup: {warmup_epochs} epochs (fixed)")
        
        self.warmup_epochs = warmup_epochs
        
        # LR scheduler
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, config.num_epochs - warmup_epochs),
            eta_min=1e-6,
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )
        
        # AMP
        self.use_amp = config.use_amp
        self.scaler = GradScaler('cuda') if self.use_amp else None
        
        # Tracking
        self.best_metric = 0.0
        self.best_epoch = 0
        self.history = []  # Per-epoch training history for plotting
        
        # Checkpoint dir
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save_history(self):
        """Save training history to JSON for plotting."""
        if not self.history:
            return
        
        history_path = self.checkpoint_dir / 'training_history.json'
        
        # Convert history list to column-oriented format for easier plotting
        columns = {}
        for entry in self.history:
            for key, value in entry.items():
                if key not in columns:
                    columns[key] = []
                columns[key].append(value)
        
        with open(history_path, 'w') as f:
            json.dump({
                'format': 'column',
                'num_epochs': len(self.history),
                'columns': columns,
                'entries': self.history,  # Also keep row format for flexibility
            }, f, indent=2)
    
    def add_history_entry(
        self,
        epoch: int,
        train_losses: Dict[str, float],
        epoch_time: float,
        val_metrics: Optional[Dict[str, float]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ):
        """Add an entry to training history."""
        entry = {
            'epoch': epoch,
            'epoch_time': round(epoch_time, 2),
            'learning_rate': self.optimizer.param_groups[0]['lr'],
        }
        
        # Add training losses with 'train_' prefix
        for k, v in train_losses.items():
            entry[f'train_{k}'] = round(v, 6)
        
        # Add validation metrics with 'val_' prefix (if provided)
        if val_metrics:
            for k, v in val_metrics.items():
                if isinstance(v, (int, float)):
                    entry[f'val_{k}'] = round(v, 6) if isinstance(v, float) else v
        
        # Add any extra info
        if extra:
            entry.update(extra)
        
        self.history.append(entry)
        self.save_history()
    
    def train_epoch(self, train_loader) -> Dict[str, float]:
        """Train for one epoch with optional gradient clipping."""
        self.model.train()
        
        total_losses = {}
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Training ({self.mode})", leave=False)
        
        for batch in pbar:
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with autocast('cuda'):
                    embeddings = self.model(images, return_normalized=False)
                    loss, loss_dict = self.loss_fn(embeddings, labels)
                
                self.scaler.scale(loss).backward()
                
                # Gradient clipping with AMP
                if self.grad_clip_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        max_norm=self.grad_clip_norm
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                embeddings = self.model(images, return_normalized=False)
                loss, loss_dict = self.loss_fn(embeddings, labels)
                
                loss.backward()
                
                # Gradient clipping without AMP
                if self.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        max_norm=self.grad_clip_norm
                    )
                
                self.optimizer.step()
            
            # Accumulate
            for k, v in loss_dict.items():
                total_losses[k] = total_losses.get(k, 0.0) + v
            num_batches += 1
            
            pbar.set_postfix({
                'loss': f"{loss_dict['total']:.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}",
            })
        
        return {k: v / num_batches for k, v in total_losses.items()}
    
    def validate(
        self,
        test_loader,
        name: str = "Test",
    ) -> Dict[str, float]:
        """Run validation on a single test loader."""
        return compute_reid_metrics(self.model, test_loader, self.device)
    
    def validate_star(
        self,
        gallery_loader,
        query_loader,
        use_enhancements: bool = False,
        return_embeddings: bool = False,
    ) -> Dict[str, float]:
        """
        Validate on star_dataset using gallery/query protocol.
        
        Gallery = train outing images
        Query = test outing images
        
        Args:
            gallery_loader: DataLoader for gallery (train outings)
            query_loader: DataLoader for query (test outings)
            use_enhancements: If True, use TTA and re-ranking from config
            return_embeddings: If True, also return embeddings dict for reuse
            
        Returns:
            Dict with mAP, Rank-1, Rank-5, Rank-10
            If return_embeddings=True, returns (metrics, embeddings_dict)
        """
        # Check if config enables inference enhancements
        use_tta = use_enhancements and getattr(self.config.model, 'use_tta', False)
        use_reranking = use_enhancements and getattr(self.config.model, 'use_reranking', False)
        
        if use_tta or use_reranking:
            # Use enhanced inference pipeline
            metrics = compute_reid_metrics_with_enhancements(
                model=self.model,
                gallery_loader=gallery_loader,
                query_loader=query_loader,
                device=self.device,
                use_tta=use_tta,
                use_reranking=use_reranking,
            )
            if return_embeddings:
                # Enhanced path doesn't cache embeddings currently
                return metrics, None
            return metrics
        
        # Standard validation (optimized - keep on GPU for similarity)
        self.model.eval()
        
        # Extract embeddings - keep on GPU
        gallery_embeddings = []
        gallery_labels = []
        query_embeddings = []
        query_labels = []
        
        with torch.no_grad():
            # Use autocast for faster inference
            with autocast('cuda', enabled=self.use_amp):
                for batch in tqdm(gallery_loader, desc="Gallery", leave=False):
                    images = batch['image'].to(self.device, non_blocking=True)
                    emb = self.model(images, return_normalized=True)
                    gallery_embeddings.append(emb)  # Keep on GPU
                    gallery_labels.append(batch['label'])
                
                for batch in tqdm(query_loader, desc="Query", leave=False):
                    images = batch['image'].to(self.device, non_blocking=True)
                    emb = self.model(images, return_normalized=True)
                    query_embeddings.append(emb)  # Keep on GPU
                    query_labels.append(batch['label'])
        
        # Concat on GPU
        gallery_embeddings_gpu = torch.cat(gallery_embeddings, dim=0)
        query_embeddings_gpu = torch.cat(query_embeddings, dim=0)
        gallery_labels = torch.cat(gallery_labels, dim=0).numpy()
        query_labels = torch.cat(query_labels, dim=0).numpy()
        
        # Compute similarities on GPU (much faster for large matrices)
        with torch.no_grad():
            similarities = (query_embeddings_gpu @ gallery_embeddings_gpu.T).cpu().numpy()
        
        # Move embeddings to CPU/numpy for potential reuse
        gallery_embeddings_np = gallery_embeddings_gpu.cpu().numpy()
        query_embeddings_np = query_embeddings_gpu.cpu().numpy()
        
        # Free GPU memory
        del gallery_embeddings_gpu, query_embeddings_gpu
        
        # Compute metrics (loop is fine - argsort dominates anyway)
        all_aps = []
        all_ranks = []
        skipped_queries = 0  # Track queries with no gallery matches
        
        for i in range(len(query_labels)):
            query_label = query_labels[i]
            sims = similarities[i]
            
            # Sort by similarity
            sorted_indices = np.argsort(-sims)
            sorted_labels = gallery_labels[sorted_indices]
            
            matches = sorted_labels == query_label
            
            if matches.sum() == 0:
                skipped_queries += 1
                continue
            
            # AP
            cumsum = np.cumsum(matches)
            precision = cumsum / (np.arange(len(matches)) + 1)
            ap = (precision * matches).sum() / matches.sum()
            all_aps.append(ap)
            
            # Rank
            first_match = np.where(matches)[0][0]
            all_ranks.append(first_match + 1)
        
        # Log and validate evaluation results
        total_queries = len(query_labels)
        if skipped_queries > 0:
            print(f"  ⚠️ WARNING: {skipped_queries}/{total_queries} queries skipped (no gallery matches)")
        
        if len(all_aps) == 0:
            raise ValueError(
                f"Evaluation failed: ALL {total_queries} queries were skipped! "
                f"No query identity has matching samples in gallery. "
                f"Check your data split - this indicates a bug in train/test partitioning."
            )
        
        metrics = {
            'mAP': np.mean(all_aps),
            'Rank-1': np.mean([1.0 if r <= 1 else 0.0 for r in all_ranks]),
            'Rank-5': np.mean([1.0 if r <= 5 else 0.0 for r in all_ranks]),
            'Rank-10': np.mean([1.0 if r <= 10 else 0.0 for r in all_ranks]),
            'num_valid_queries': len(all_aps),
            'num_skipped_queries': skipped_queries,
        }
        
        if return_embeddings:
            embeddings_dict = {
                'gallery_embeddings': gallery_embeddings_np,
                'gallery_labels': gallery_labels,
                'query_embeddings': query_embeddings_np,
                'query_labels': query_labels,
                'similarities': similarities,
            }
            return metrics, embeddings_dict
        
        return metrics
    
    def save_checkpoint(
        self,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False,
        suffix: str = "",
    ):
        """Save checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'mode': self.mode,
            'best_metric': self.best_metric,
        }
        
        # Latest
        name = f'latest{suffix}.pth'
        torch.save(checkpoint, self.checkpoint_dir / name)
        
        # Best
        if is_best:
            name = f'best{suffix}.pth'
            torch.save(checkpoint, self.checkpoint_dir / name)
            print(f"  💾 Saved best model (mAP: {metrics.get('mAP', 0):.4f})")
    
    def load_checkpoint(self, path: str) -> Tuple[int, Dict[str, float]]:
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_metric = checkpoint.get('best_metric', 0.0)
        return checkpoint['epoch'], checkpoint.get('metrics', {})
    
    def train_pretrain(
        self,
        train_loader,
        test_loader,
        num_epochs: Optional[int] = None,
        val_every: int = 5,
        validate_first: bool = True,
    ) -> Dict[str, float]:
        """Pre-training loop on Wildlife10k."""
        num_epochs = num_epochs or self.config.num_epochs
        
        print(f"\n{'='*60}")
        print(f"PRE-TRAINING ON WILDLIFE10K")
        print(f"{'='*60}")
        print(f"Epochs: {num_epochs}, Val every: {val_every}")
        print(f"Checkpoints: {self.checkpoint_dir}")
        
        best_metrics = {}
        
        # Initial validation (epoch 0 baseline)
        if validate_first:
            print("  [Epoch 0] Initial validation on Wildlife10k...")
            metrics = self.validate(test_loader)
            metric_str = " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
            print(f"  {metric_str}")
            self.best_metric = metrics['mAP']
            best_metrics = metrics.copy()
        
        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            
            train_losses = self.train_epoch(train_loader)
            self.scheduler.step()
            
            epoch_time = time.time() - epoch_start
            loss_str = " | ".join(f"{k}: {v:.4f}" for k, v in train_losses.items())
            print(f"Epoch {epoch}/{num_epochs} ({epoch_time:.1f}s) | {loss_str}")
            
            # Validate
            val_metrics = None
            if epoch % val_every == 0 or epoch == num_epochs:
                print("  Validating on Wildlife10k...")
                val_metrics = self.validate(test_loader)
                metric_str = " | ".join(f"{k}: {v:.4f}" for k, v in val_metrics.items())
                print(f"  {metric_str}")
                
                is_best = val_metrics['mAP'] > self.best_metric
                if is_best:
                    self.best_metric = val_metrics['mAP']
                    self.best_epoch = epoch
                    best_metrics = val_metrics.copy()
                
                self.save_checkpoint(epoch, val_metrics, is_best)
            
            # Log history
            self.add_history_entry(epoch, train_losses, epoch_time, val_metrics)
        
        print(f"\n✅ Pre-training complete! Best mAP: {self.best_metric:.4f} (epoch {self.best_epoch})")
        return best_metrics
    
    def train_finetune(
        self,
        train_loader,
        gallery_loader,
        query_loader,
        num_epochs: Optional[int] = None,
        val_every: int = 5,
        validate_first: bool = True,
    ) -> Dict[str, float]:
        """Fine-tuning loop on star_dataset."""
        num_epochs = num_epochs or self.config.num_epochs
        
        print(f"\n{'='*60}")
        print(f"FINE-TUNING ON STAR_DATASET")
        print(f"{'='*60}")
        print(f"Epochs: {num_epochs}, Val every: {val_every}")
        print(f"Checkpoints: {self.checkpoint_dir}")
        
        best_metrics = {}
        
        # Initial validation (epoch 0 baseline)
        if validate_first:
            print("  [Epoch 0] Initial validation on star_dataset...")
            metrics = self.validate_star(gallery_loader, query_loader)
            metric_str = " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
            print(f"  {metric_str}")
            self.best_metric = metrics['mAP']
            best_metrics = metrics.copy()
        
        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            
            train_losses = self.train_epoch(train_loader)
            self.scheduler.step()
            
            epoch_time = time.time() - epoch_start
            loss_str = " | ".join(f"{k}: {v:.4f}" for k, v in train_losses.items())
            print(f"Epoch {epoch}/{num_epochs} ({epoch_time:.1f}s) | {loss_str}")
            
            # Validate
            val_metrics = None
            if epoch % val_every == 0 or epoch == num_epochs:
                print("  Validating on star_dataset...")
                val_metrics = self.validate_star(gallery_loader, query_loader)
                metric_str = " | ".join(f"{k}: {v:.4f}" for k, v in val_metrics.items())
                print(f"  {metric_str}")
                
                is_best = val_metrics['mAP'] > self.best_metric
                if is_best:
                    self.best_metric = val_metrics['mAP']
                    self.best_epoch = epoch
                    best_metrics = val_metrics.copy()
                
                self.save_checkpoint(epoch, val_metrics, is_best)
            
            # Log history
            self.add_history_entry(epoch, train_losses, epoch_time, val_metrics)
        
        print(f"\n✅ Fine-tuning complete! Best mAP: {self.best_metric:.4f} (epoch {self.best_epoch})")
        
        # Run detailed per-identity/per-folder evaluation with full inference enhancements
        # TTA + reranking by default for final evaluation (more accurate but slower)
        print("\nRunning detailed star_dataset evaluation (with TTA + reranking)...")
        from megastarid.evaluation import compute_detailed_star_metrics
        
        detailed_metrics = compute_detailed_star_metrics(
            model=self.model,
            gallery_loader=gallery_loader,
            query_loader=query_loader,
            device=self.device,
            output_dir=self.checkpoint_dir,
            use_tta=True,
            use_reranking=True,
        )
        
        return best_metrics
    
    def train_cotrain(
        self,
        train_loader,
        wildlife_test_loader,
        star_gallery_loader,
        star_query_loader,
        num_epochs: Optional[int] = None,
        val_every: int = 5,
        validate_first: bool = True,
    ) -> Dict[str, Any]:
        """Co-training loop on Wildlife10k + star_dataset."""
        num_epochs = num_epochs or self.config.num_epochs
        
        print(f"\n{'='*60}")
        print(f"CO-TRAINING ON WILDLIFE10K + STAR_DATASET")
        print(f"{'='*60}")
        print(f"Epochs: {num_epochs}, Val every: {val_every}")
        print(f"Checkpoints: {self.checkpoint_dir}")
        
        best_star_metrics = {}
        
        # Initial validation (epoch 0 baseline)
        if validate_first:
            print("  [Epoch 0] Initial validation...")
            wildlife_metrics = self.validate(wildlife_test_loader)
            print(f"  Wildlife: " + " | ".join(f"{k}: {v:.4f}" for k, v in wildlife_metrics.items()))
            star_metrics = self.validate_star(star_gallery_loader, star_query_loader)
            print(f"  Star: " + " | ".join(f"{k}: {v:.4f}" for k, v in star_metrics.items()))
            self.best_metric = star_metrics['mAP']
            best_star_metrics = star_metrics.copy()
        
        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            
            train_losses = self.train_epoch(train_loader)
            self.scheduler.step()
            
            epoch_time = time.time() - epoch_start
            loss_str = " | ".join(f"{k}: {v:.4f}" for k, v in train_losses.items())
            print(f"Epoch {epoch}/{num_epochs} ({epoch_time:.1f}s) | {loss_str}")
            
            # Validate
            val_metrics = None
            if epoch % val_every == 0 or epoch == num_epochs:
                # Wildlife10k metrics
                print("  Validating on Wildlife10k...")
                wildlife_metrics = self.validate(wildlife_test_loader)
                print(f"  Wildlife: " + " | ".join(f"{k}: {v:.4f}" for k, v in wildlife_metrics.items()))
                
                # Star metrics (primary)
                print("  Validating on star_dataset...")
                star_metrics = self.validate_star(star_gallery_loader, star_query_loader)
                print(f"  Star: " + " | ".join(f"{k}: {v:.4f}" for k, v in star_metrics.items()))
                
                # Use star mAP as primary metric
                is_best = star_metrics['mAP'] > self.best_metric
                if is_best:
                    self.best_metric = star_metrics['mAP']
                    self.best_epoch = epoch
                    best_star_metrics = star_metrics.copy()
                
                combined_metrics = {
                    'star_mAP': star_metrics['mAP'],
                    'wildlife_mAP': wildlife_metrics['mAP'],
                    **{f'star_{k}': v for k, v in star_metrics.items()},
                    **{f'wildlife_{k}': v for k, v in wildlife_metrics.items()},
                }
                
                self.save_checkpoint(epoch, combined_metrics, is_best)
                
                # Combine for history (star_ and wildlife_ prefixes)
                val_metrics = {
                    **{f'star_{k}': v for k, v in star_metrics.items()},
                    **{f'wildlife_{k}': v for k, v in wildlife_metrics.items()},
                }
            
            # Log history
            self.add_history_entry(epoch, train_losses, epoch_time, val_metrics)
        
        print(f"\n✅ Co-training complete! Best star mAP: {self.best_metric:.4f} (epoch {self.best_epoch})")
        
        # Run detailed per-identity/per-folder evaluation with full inference enhancements
        # TTA + reranking by default for final evaluation (more accurate but slower)
        print("\nRunning detailed star_dataset evaluation (with TTA + reranking)...")
        from megastarid.evaluation import compute_detailed_star_metrics
        
        detailed_metrics = compute_detailed_star_metrics(
            model=self.model,
            gallery_loader=star_gallery_loader,
            query_loader=star_query_loader,
            device=self.device,
            output_dir=self.checkpoint_dir,
            use_tta=True,
            use_reranking=True,
        )
        
        return best_star_metrics

