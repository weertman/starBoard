#!/usr/bin/env python
"""
Architecture Grid Search for MegaStarID.

Tests combinations of:
- Model architecture: swinv2-tiny, densenet121, densenet169
- Training strategy: pretrain→finetune, cotrain, star-only (baseline)
- Loss function: circle-only, triplet-only

Optimizes batch size per architecture for dual A6000 GPUs (40GB usable each).

Usage:
    python -m megastarid.experiments.arch_grid_search --epochs 10 --gpus 0,1
    python -m megastarid.experiments.arch_grid_search --dry-run
    python -m megastarid.experiments.arch_grid_search --architectures densenet121 densenet169
"""
import argparse
import json
import time
import random
import gc
import platform
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Windows multiprocessing fix: spawn method is slow and can deadlock with many workers
IS_WINDOWS = platform.system() == 'Windows'
MAX_WORKERS_WINDOWS = 8  # Windows worker limit

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from megastarid.config import (
    PretrainConfig, FinetuneConfig, CotrainConfig,
    ModelConfig, LossConfig
)
from megastarid.datasets import (
    create_pretrain_dataloaders,
    create_finetune_dataloaders,
    create_cotrain_dataloaders,
)
from megastarid.models import create_model, load_pretrained_model, count_parameters, get_model_info
from megastarid.trainer import MegaStarTrainer
from megastarid.experiments.plotting import (
    plot_training_history,
    plot_grid_search_summary,
    ExperimentResult,
)


# =============================================================================
# Batch Size Optimization for A6000 GPUs (48GB each, using ~35-40GB)
# =============================================================================

# Batch sizes optimized for single A6000 (48GB VRAM)
# These are per-GPU batch sizes - with 2 GPUs, total = 2x these values
# Tested with 384x384 images + AMP + gradients + optimizer states
BATCH_SIZE_MAP = {
    # Architecture: (batch_size_per_gpu, num_instances)
    
    # SwinV2-Tiny: 29M params, ~12GB at batch 80
    # Single A6000 can do 160-200 easily
    'swinv2-tiny': (160, 4),  # Single GPU: 160, Dual: 320
    
    # DenseNet121: 8.7M params, very memory efficient
    # Can go very large on single A6000
    'densenet121': (320, 4),  # Single GPU: 320, Dual: 640
    
    # DenseNet169: 14M params
    'densenet169': (256, 4),  # Single GPU: 256, Dual: 512
    
    # ResNet50: 25M params, literature baseline
    # Very well-optimized architecture
    'resnet50': (256, 4),  # Single GPU: 256, Dual: 512
    
    # ConvNeXt-Tiny: 28M params, modern CNN
    # Similar memory profile to SwinV2-Tiny
    'convnext-tiny': (160, 4),  # Single GPU: 160, Dual: 320
}


# =============================================================================
# Architecture-Specific Hyperparameters
# =============================================================================
# Based on best practices for each architecture type:
# - Transformers (SwinV2) need lower LR, higher weight decay, layer-wise LR decay
# - CNNs (DenseNet) can handle higher LR, lower weight decay

HYPERPARAM_MAP = {
    'swinv2-tiny': {
        'learning_rate': 2e-4,         # Higher LR for small dataset fine-tuning
        'weight_decay': 1e-3,          # Much lower WD for small datasets (was 0.02 - too aggressive)
        'warmup_ratio': 0.10,          # 10% warmup like CNNs
        'grad_clip_norm': 1.0,         # Gradient clipping for stability
        'use_llrd': True,              # Layer-wise learning rate decay
        'llrd_decay': 0.9,             # Gentler decay (was 0.75 - too aggressive for small datasets)
        'backbone_lr_mult': 0.1,       # Backbone at 10% of head LR (if not using LLRD)
    },
    'densenet121': {
        'learning_rate': 2e-4,         # CNNs can handle higher LR
        'weight_decay': 1e-4,          # Standard weight decay for CNNs
        'warmup_ratio': 0.10,          # 10% warmup
        'grad_clip_norm': 5.0,         # Less aggressive clipping for CNNs
        'use_llrd': False,             # No LLRD for CNNs
        'llrd_decay': 1.0,
        'backbone_lr_mult': 0.1,       # Backbone at 10% of head LR
    },
    'densenet169': {
        'learning_rate': 1.5e-4,       # Slightly lower than densenet121 (more params)
        'weight_decay': 1e-4,
        'warmup_ratio': 0.10,
        'grad_clip_norm': 5.0,
        'use_llrd': False,
        'llrd_decay': 1.0,
        'backbone_lr_mult': 0.1,
    },
    'resnet50': {
        'learning_rate': 2e-4,         # Standard LR for ResNet
        'weight_decay': 5e-4,          # ResNet typically uses higher weight decay
        'warmup_ratio': 0.10,          # 10% warmup
        'grad_clip_norm': 5.0,         # Standard clipping
        'use_llrd': False,             # No LLRD for CNNs
        'llrd_decay': 1.0,
        'backbone_lr_mult': 0.1,       # Backbone at 10% of head LR
    },
    'convnext-tiny': {
        'learning_rate': 2e-4,         # Higher LR for small dataset fine-tuning
        'weight_decay': 1e-3,          # Much lower WD for small datasets (was 0.05 - too aggressive)
        'warmup_ratio': 0.10,          # 10% warmup
        'grad_clip_norm': 1.0,         # ConvNeXt benefits from clipping
        'use_llrd': True,              # ConvNeXt benefits from LLRD (transformer-like training)
        'llrd_decay': 0.9,             # Gentler decay for small datasets
        'backbone_lr_mult': 0.1,       # Backbone at 10% of head LR (if not using LLRD)
    },
}


def get_arch_hyperparams(architecture: str) -> dict:
    """Get architecture-specific hyperparameters."""
    if architecture not in HYPERPARAM_MAP:
        print(f"Warning: Unknown architecture {architecture}, using swinv2-tiny defaults")
        return HYPERPARAM_MAP['swinv2-tiny']
    return HYPERPARAM_MAP[architecture].copy()

# DataLoader optimization settings
DATALOADER_CONFIG = {
    'num_workers': 16,  # Use many workers for fast SSD
    'prefetch_factor': 4,  # Prefetch more batches
    'persistent_workers': True,  # Keep workers alive between epochs
    'pin_memory': True,  # Faster GPU transfer
}


def get_batch_config(
    architecture: str, 
    num_gpus: int = 2,
    max_batch_size: Optional[int] = None,
) -> Tuple[int, int]:
    """
    Get optimal batch size and num_instances for architecture.
    
    Args:
        architecture: Model backbone name
        num_gpus: Number of GPUs (for DataParallel)
        max_batch_size: Optional cap on batch size (useful for small datasets)
        
    Returns:
        (total_batch_size, num_instances)
    """
    if architecture not in BATCH_SIZE_MAP:
        # Fallback for unknown architectures
        print(f"Warning: Unknown architecture {architecture}, using conservative batch size")
        return (32, 4)
    
    per_gpu, num_instances = BATCH_SIZE_MAP[architecture]
    total_batch = per_gpu * num_gpus
    
    # Apply max batch size cap if specified
    if max_batch_size is not None and total_batch > max_batch_size:
        total_batch = max_batch_size
    
    # Ensure batch_size is divisible by num_instances for PK sampling
    total_batch = (total_batch // num_instances) * num_instances
    
    return (total_batch, num_instances)


# =============================================================================
# Experiment Configuration
# =============================================================================

@dataclass
class ArchExperimentConfig:
    """Configuration for a single architecture experiment."""
    name: str
    architecture: str  # 'swinv2-tiny', 'densenet121', 'densenet169'
    strategy: str  # 'pretrain_finetune', 'cotrain', 'star_only'
    loss_type: str  # 'circle', 'triplet'
    
    # Epochs (will be set by command line)
    pretrain_epochs: int = 10
    finetune_epochs: int = 10
    cotrain_epochs: int = 10
    
    # Batch settings (auto-configured based on architecture)
    batch_size: int = 32
    num_instances: int = 4
    
    # For cotrain
    star_batch_ratio: float = 0.3
    
    # Model config options
    use_multiscale: bool = True
    use_bnneck: bool = True
    embedding_dim: int = 512
    
    # Architecture-specific hyperparameters (auto-configured)
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    warmup_ratio: float = 0.1          # Fraction of epochs for warmup
    grad_clip_norm: float = 1.0        # Max gradient norm
    use_llrd: bool = False             # Layer-wise learning rate decay
    llrd_decay: float = 0.75           # LLRD decay factor per layer
    backbone_lr_mult: float = 0.1      # Backbone LR multiplier (if not using LLRD)
    
    # Whether to include negative-only identities (single-outing stars) in training
    include_negative_only: bool = False


@dataclass 
class ArchExperimentResult:
    """Results from a single architecture experiment."""
    name: str
    architecture: str
    strategy: str
    loss_type: str
    
    # Model info
    total_params_millions: float
    batch_size: int
    
    # Final metrics on star_dataset
    star_mAP: float
    star_rank1: float
    star_rank5: float
    
    # Training info
    total_time_seconds: float
    best_epoch: int
    
    # Hyperparameters used
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    use_llrd: bool = False
    include_negative_only: bool = False
    
    # All metrics
    all_metrics: Dict[str, float] = None


def set_seed(seed: int):
    """Set random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_loss_config(loss_type: str) -> LossConfig:
    """Get loss config for circle-only or triplet-only."""
    if loss_type == 'circle':
        return LossConfig(circle_weight=1.0, triplet_weight=0.0)
    elif loss_type == 'triplet':
        return LossConfig(circle_weight=0.0, triplet_weight=1.0)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def get_model_config(
    architecture: str,
    use_multiscale: bool = True,
    use_bnneck: bool = True,
    embedding_dim: int = 512,
) -> ModelConfig:
    """Create ModelConfig for the specified architecture."""
    return ModelConfig(
        backbone=architecture,
        use_multiscale=use_multiscale,
        use_bnneck=use_bnneck,
        embedding_dim=embedding_dim,
        embedding_head_depth=3,
        use_residual_head=False,
        use_attention_pooling=False,
        image_size=384,
        pretrained=True,
    )


def clear_gpu_memory():
    """Clear GPU memory between experiments."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# =============================================================================
# Training Functions
# =============================================================================

def run_star_only(
    exp: ArchExperimentConfig,
    gpu_ids: List[int],
    base_dir: Path,
    seed: int = 42,
    num_workers: int = 4,
) -> ArchExperimentResult:
    """Run star-only baseline experiment."""
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {exp.name}")
    print(f"Architecture: {exp.architecture} | Strategy: star-only | Loss: {exp.loss_type}")
    print(f"Batch size: {exp.batch_size} | Epochs: {exp.finetune_epochs}")
    print(f"LR: {exp.learning_rate:.2e} | Weight decay: {exp.weight_decay:.2e} | LLRD: {exp.use_llrd}")
    print(f"Include negative-only IDs: {exp.include_negative_only}")
    print(f"{'='*70}")
    
    clear_gpu_memory()
    start_time = time.time()
    device = torch.device(f'cuda:{gpu_ids[0]}' if torch.cuda.is_available() else 'cpu')
    
    exp_dir = base_dir / exp.name
    
    config = FinetuneConfig(
        checkpoint_dir=str(exp_dir),
        pretrain_checkpoint=None,
        model=get_model_config(exp.architecture, exp.use_multiscale, exp.use_bnneck),
        loss=get_loss_config(exp.loss_type),
        num_epochs=exp.finetune_epochs,
        batch_size=exp.batch_size,
        num_instances=exp.num_instances,
        # Architecture-specific hyperparameters
        learning_rate=exp.learning_rate,
        weight_decay=exp.weight_decay,
        warmup_ratio=exp.warmup_ratio,
        grad_clip_norm=exp.grad_clip_norm,
        use_llrd=exp.use_llrd,
        llrd_decay=exp.llrd_decay,
        backbone_lr_mult=exp.backbone_lr_mult,
        num_workers=num_workers,
        seed=seed,
        val_every_n_epochs=max(1, exp.finetune_epochs // 5),
        # Negative-only identities experiment
        include_negative_only=exp.include_negative_only,
    )
    
    # Create model
    model = create_model(config)
    model_info = get_model_info(model)
    print(f"Model params: {model_info['total_params_millions']:.1f}M")
    
    if len(gpu_ids) > 1:
        print(f"Using DataParallel on GPUs: {gpu_ids}")
        model = nn.DataParallel(model, device_ids=gpu_ids)
    model = model.to(device)
    
    # Load data
    train_loader, gallery_loader, query_loader = create_finetune_dataloaders(config)
    
    # Train
    trainer = MegaStarTrainer(model, config, device, mode='finetune')
    best_metrics = trainer.train_finetune(
        train_loader, gallery_loader, query_loader,
        exp.finetune_epochs, val_every=5, validate_first=True
    )
    
    total_time = time.time() - start_time
    
    return ArchExperimentResult(
        name=exp.name,
        architecture=exp.architecture,
        strategy='star_only',
        loss_type=exp.loss_type,
        total_params_millions=model_info['total_params_millions'],
        batch_size=exp.batch_size,
        star_mAP=best_metrics.get('mAP', 0),
        star_rank1=best_metrics.get('Rank-1', 0),
        star_rank5=best_metrics.get('Rank-5', 0),
        total_time_seconds=total_time,
        best_epoch=trainer.best_epoch,
        learning_rate=exp.learning_rate,
        weight_decay=exp.weight_decay,
        use_llrd=exp.use_llrd,
        include_negative_only=exp.include_negative_only,
        all_metrics=best_metrics,
    )


def run_pretrain_finetune(
    exp: ArchExperimentConfig,
    gpu_ids: List[int],
    base_dir: Path,
    seed: int = 42,
    num_workers: int = 4,
) -> ArchExperimentResult:
    """Run pretrain → finetune experiment."""
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {exp.name}")
    print(f"Architecture: {exp.architecture} | Strategy: pretrain→finetune | Loss: {exp.loss_type}")
    print(f"Batch size: {exp.batch_size} | Epochs: {exp.pretrain_epochs}+{exp.finetune_epochs}")
    print(f"LR: {exp.learning_rate:.2e} | Weight decay: {exp.weight_decay:.2e} | LLRD: {exp.use_llrd}")
    print(f"{'='*70}")
    
    clear_gpu_memory()
    start_time = time.time()
    device = torch.device(f'cuda:{gpu_ids[0]}' if torch.cuda.is_available() else 'cpu')
    
    exp_dir = base_dir / exp.name
    # Note: PretrainConfig.__post_init__ appends "pretrain" to checkpoint_dir
    # So we pass exp_dir, not exp_dir/'pretrain'
    finetune_dir = exp_dir / 'finetune'
    
    model_config = get_model_config(exp.architecture, exp.use_multiscale, exp.use_bnneck)
    
    # ===== PHASE 1: Pre-training on Wildlife10k =====
    print("\n--- Phase 1: Pre-training on Wildlife10k ---")
    
    pretrain_config = PretrainConfig(
        checkpoint_dir=str(exp_dir),  # PretrainConfig adds /pretrain automatically
        model=model_config,
        loss=get_loss_config(exp.loss_type),
        num_epochs=exp.pretrain_epochs,
        batch_size=exp.batch_size,
        num_instances=exp.num_instances,
        # Architecture-specific hyperparameters
        learning_rate=exp.learning_rate,
        weight_decay=exp.weight_decay,
        warmup_ratio=exp.warmup_ratio,
        grad_clip_norm=exp.grad_clip_norm,
        use_llrd=exp.use_llrd,
        llrd_decay=exp.llrd_decay,
        backbone_lr_mult=exp.backbone_lr_mult,
        num_workers=num_workers,
        seed=seed,
        exclude_datasets=['SeaStarReID2023'],
        val_every_n_epochs=max(1, exp.pretrain_epochs // 3),
    )
    
    train_loader, test_loader = create_pretrain_dataloaders(pretrain_config)
    
    model = create_model(pretrain_config)
    model_info = get_model_info(model)
    print(f"Model params: {model_info['total_params_millions']:.1f}M")
    
    if len(gpu_ids) > 1:
        print(f"Using DataParallel on GPUs: {gpu_ids}")
        model = nn.DataParallel(model, device_ids=gpu_ids)
    model = model.to(device)
    
    trainer = MegaStarTrainer(model, pretrain_config, device, mode='pretrain')
    trainer.train_pretrain(train_loader, test_loader, exp.pretrain_epochs, 
                          val_every=5, validate_first=True)
    
    # PretrainConfig adds /pretrain to exp_dir, so checkpoint is at exp_dir/pretrain/best.pth
    pretrain_checkpoint = exp_dir / 'pretrain' / 'best.pth'
    
    # Clean up pretrain model
    del model, trainer
    clear_gpu_memory()
    
    # ===== PHASE 2: Fine-tuning on star_dataset =====
    print("\n--- Phase 2: Fine-tuning on star_dataset ---")
    
    # Use lower LR for finetuning (half of pretrain LR)
    finetune_lr = exp.learning_rate * 0.5
    
    finetune_config = FinetuneConfig(
        checkpoint_dir=str(finetune_dir),
        pretrain_checkpoint=str(pretrain_checkpoint),
        model=model_config,
        loss=get_loss_config(exp.loss_type),
        num_epochs=exp.finetune_epochs,
        batch_size=exp.batch_size,
        num_instances=exp.num_instances,
        # Architecture-specific hyperparameters (lower LR for finetuning)
        learning_rate=finetune_lr,
        weight_decay=exp.weight_decay,
        warmup_ratio=exp.warmup_ratio,
        grad_clip_norm=exp.grad_clip_norm,
        use_llrd=exp.use_llrd,
        llrd_decay=exp.llrd_decay,
        backbone_lr_mult=exp.backbone_lr_mult,
        num_workers=num_workers,
        seed=seed,
        val_every_n_epochs=max(1, exp.finetune_epochs // 3),
    )
    
    train_loader, gallery_loader, query_loader = create_finetune_dataloaders(finetune_config)
    
    model = load_pretrained_model(finetune_config, str(pretrain_checkpoint), device)
    if len(gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=gpu_ids)
    model = model.to(device)
    
    trainer = MegaStarTrainer(model, finetune_config, device, mode='finetune')
    best_metrics = trainer.train_finetune(
        train_loader, gallery_loader, query_loader,
        exp.finetune_epochs, val_every=5, validate_first=True
    )
    
    total_time = time.time() - start_time
    
    return ArchExperimentResult(
        name=exp.name,
        architecture=exp.architecture,
        strategy='pretrain_finetune',
        loss_type=exp.loss_type,
        total_params_millions=model_info['total_params_millions'],
        batch_size=exp.batch_size,
        star_mAP=best_metrics.get('mAP', 0),
        star_rank1=best_metrics.get('Rank-1', 0),
        star_rank5=best_metrics.get('Rank-5', 0),
        total_time_seconds=total_time,
        best_epoch=trainer.best_epoch,
        learning_rate=exp.learning_rate,
        weight_decay=exp.weight_decay,
        use_llrd=exp.use_llrd,
        all_metrics=best_metrics,
    )


def run_cotrain(
    exp: ArchExperimentConfig,
    gpu_ids: List[int],
    base_dir: Path,
    seed: int = 42,
    num_workers: int = 4,
) -> ArchExperimentResult:
    """Run co-training experiment."""
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {exp.name}")
    print(f"Architecture: {exp.architecture} | Strategy: cotrain | Loss: {exp.loss_type}")
    print(f"Batch size: {exp.batch_size} | Epochs: {exp.cotrain_epochs}")
    print(f"LR: {exp.learning_rate:.2e} | Weight decay: {exp.weight_decay:.2e} | LLRD: {exp.use_llrd}")
    print(f"{'='*70}")
    
    clear_gpu_memory()
    start_time = time.time()
    device = torch.device(f'cuda:{gpu_ids[0]}' if torch.cuda.is_available() else 'cpu')
    
    exp_dir = base_dir / exp.name
    
    config = CotrainConfig(
        checkpoint_dir=str(exp_dir),
        model=get_model_config(exp.architecture, exp.use_multiscale, exp.use_bnneck),
        loss=get_loss_config(exp.loss_type),
        num_epochs=exp.cotrain_epochs,
        batch_size=exp.batch_size,
        num_instances=exp.num_instances,
        star_batch_ratio=exp.star_batch_ratio,
        # Architecture-specific hyperparameters
        learning_rate=exp.learning_rate,
        weight_decay=exp.weight_decay,
        warmup_ratio=exp.warmup_ratio,
        grad_clip_norm=exp.grad_clip_norm,
        use_llrd=exp.use_llrd,
        llrd_decay=exp.llrd_decay,
        backbone_lr_mult=exp.backbone_lr_mult,
        num_workers=num_workers,
        seed=seed,
        exclude_datasets=['SeaStarReID2023'],
        val_every_n_epochs=max(1, exp.cotrain_epochs // 3),
    )
    
    train_loader, wildlife_test, star_gallery, star_query = create_cotrain_dataloaders(config)
    
    model = create_model(config)
    model_info = get_model_info(model)
    print(f"Model params: {model_info['total_params_millions']:.1f}M")
    
    if len(gpu_ids) > 1:
        print(f"Using DataParallel on GPUs: {gpu_ids}")
        model = nn.DataParallel(model, device_ids=gpu_ids)
    model = model.to(device)
    
    trainer = MegaStarTrainer(model, config, device, mode='cotrain')
    best_metrics = trainer.train_cotrain(
        train_loader, wildlife_test, star_gallery, star_query,
        exp.cotrain_epochs, val_every=5, validate_first=True
    )
    
    total_time = time.time() - start_time
    
    return ArchExperimentResult(
        name=exp.name,
        architecture=exp.architecture,
        strategy='cotrain',
        loss_type=exp.loss_type,
        total_params_millions=model_info['total_params_millions'],
        batch_size=exp.batch_size,
        star_mAP=best_metrics.get('mAP', 0),
        star_rank1=best_metrics.get('Rank-1', 0),
        star_rank5=best_metrics.get('Rank-5', 0),
        total_time_seconds=total_time,
        best_epoch=trainer.best_epoch,
        learning_rate=exp.learning_rate,
        weight_decay=exp.weight_decay,
        use_llrd=exp.use_llrd,
        all_metrics=best_metrics,
    )


def run_experiment(
    exp: ArchExperimentConfig,
    gpu_ids: List[int],
    base_dir: Path,
    seed: int,
    num_workers: int,
    max_retries: int = 3,
) -> ArchExperimentResult:
    """
    Run a single experiment based on its strategy.
    
    Automatically retries with reduced batch size on OOM.
    """
    original_batch_size = exp.batch_size
    
    for attempt in range(max_retries):
        try:
            clear_gpu_memory()
            
            if exp.strategy == 'pretrain_finetune':
                return run_pretrain_finetune(exp, gpu_ids, base_dir, seed, num_workers)
            elif exp.strategy == 'cotrain':
                return run_cotrain(exp, gpu_ids, base_dir, seed, num_workers)
            elif exp.strategy == 'star_only':
                return run_star_only(exp, gpu_ids, base_dir, seed, num_workers)
            else:
                raise ValueError(f"Unknown strategy: {exp.strategy}")
                
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "CUDA out of memory" in str(e):
                clear_gpu_memory()
                
                # Reduce batch size by 25%
                new_batch_size = int(exp.batch_size * 0.75)
                # Round down to multiple of num_instances
                new_batch_size = (new_batch_size // exp.num_instances) * exp.num_instances
                new_batch_size = max(new_batch_size, exp.num_instances * 2)  # Minimum viable
                
                if attempt < max_retries - 1:
                    print(f"\n⚠️  OOM Error! Reducing batch size: {exp.batch_size} → {new_batch_size}")
                    print(f"    Retry {attempt + 2}/{max_retries}...")
                    exp.batch_size = new_batch_size
                else:
                    print(f"\n❌ OOM Error after {max_retries} attempts. Skipping experiment.")
                    raise
            else:
                # Not an OOM error, re-raise
                raise
    
    # Should not reach here
    raise RuntimeError("Max retries exceeded")


# =============================================================================
# Experiment Grid Creation
# =============================================================================

def create_experiment_grid(
    epochs: int = 10,
    architectures: List[str] = None,
    strategies: List[str] = None,
    loss_types: List[str] = None,
    num_gpus: int = 2,
    max_batch_size: Optional[int] = None,
    include_negative_only_options: List[bool] = None,
) -> List[ArchExperimentConfig]:
    """
    Create grid of experiments to run.
    
    Args:
        epochs: Epochs per training phase
        architectures: List of architectures to test
        strategies: List of strategies to test
        loss_types: List of loss types to test
        num_gpus: Number of GPUs for batch size calculation
        max_batch_size: Optional cap on batch size (for small datasets)
        include_negative_only_options: List of [False, True] to test both settings
        
    Returns:
        List of experiment configurations
    """
    if architectures is None:
        architectures = ['swinv2-tiny', 'densenet121', 'densenet169']
    if strategies is None:
        strategies = ['star_only', 'pretrain_finetune', 'cotrain']
    if loss_types is None:
        loss_types = ['circle', 'triplet']
    if include_negative_only_options is None:
        include_negative_only_options = [False]  # Default: don't include
    
    experiments = []
    
    # For star_only strategy, we have limited identities (~67 valid with 4 instances each)
    # Max batch = 67 * 4 = 268, so cap at 256 to be safe
    STAR_ONLY_MAX_BATCH = 256
    
    for arch in architectures:
        hyperparams = get_arch_hyperparams(arch)
        
        for strategy in strategies:
            # Apply strategy-specific batch size cap
            effective_max_batch = max_batch_size
            if strategy == 'star_only':
                if effective_max_batch is None or effective_max_batch > STAR_ONLY_MAX_BATCH:
                    effective_max_batch = STAR_ONLY_MAX_BATCH
            
            batch_size, num_instances = get_batch_config(arch, num_gpus, effective_max_batch)
            
            for loss in loss_types:
                for include_neg in include_negative_only_options:
                    # Build experiment name
                    name = f"{arch}_{strategy}_{loss}"
                    if include_neg:
                        name += "_negonly"
                    
                    experiments.append(ArchExperimentConfig(
                        name=name,
                        architecture=arch,
                        strategy=strategy,
                        loss_type=loss,
                        pretrain_epochs=epochs,
                        finetune_epochs=epochs,
                        cotrain_epochs=epochs,
                        batch_size=batch_size,
                        num_instances=num_instances,
                        # Architecture-specific hyperparameters
                        learning_rate=hyperparams['learning_rate'],
                        weight_decay=hyperparams['weight_decay'],
                        warmup_ratio=hyperparams['warmup_ratio'],
                        grad_clip_norm=hyperparams['grad_clip_norm'],
                        use_llrd=hyperparams['use_llrd'],
                        llrd_decay=hyperparams['llrd_decay'],
                        backbone_lr_mult=hyperparams['backbone_lr_mult'],
                        # Negative-only experiment
                        include_negative_only=include_neg,
                    ))
    
    return experiments


# =============================================================================
# Results Management
# =============================================================================

def save_results(results: List[ArchExperimentResult], output_path: Path):
    """Save results to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        'timestamp': datetime.now().isoformat(),
        'num_experiments': len(results),
        'results': [asdict(r) for r in results],
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


def print_summary(results: List[ArchExperimentResult]):
    """Print summary table of results."""
    print("\n" + "=" * 100)
    print("ARCHITECTURE GRID SEARCH RESULTS")
    print("=" * 100)
    print(f"{'Experiment':<45} {'Arch':<12} {'Params':>8} {'Batch':>6} {'mAP':>8} {'R1':>8} {'Time':>10}")
    print("-" * 100)
    
    # Sort by mAP
    sorted_results = sorted(results, key=lambda x: x.star_mAP, reverse=True)
    
    for r in sorted_results:
        time_str = f"{r.total_time_seconds/60:.1f}m"
        arch_short = r.architecture.replace('densenet', 'dn').replace('swinv2-', 'sv2-')
        print(f"{r.name:<45} {arch_short:<12} {r.total_params_millions:>7.1f}M {r.batch_size:>6} {r.star_mAP:>8.4f} {r.star_rank1:>8.4f} {time_str:>10}")
    
    print("-" * 100)
    
    # Best per architecture
    print("\n🏆 Best per Architecture:")
    for arch in ['swinv2-tiny', 'densenet121', 'densenet169', 'resnet50', 'convnext-tiny']:
        arch_results = [r for r in sorted_results if r.architecture == arch]
        if arch_results:
            best = arch_results[0]
            print(f"  {arch}: {best.name} (mAP={best.star_mAP:.4f})")
    
    # Overall best
    if sorted_results:
        best = sorted_results[0]
        print(f"\n🥇 Overall Best: {best.name}")
        print(f"   Architecture: {best.architecture} ({best.total_params_millions:.1f}M params)")
        print(f"   mAP: {best.star_mAP:.4f} | Rank-1: {best.star_rank1:.4f}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Architecture grid search experiments')
    
    parser.add_argument('--epochs', type=int, default=10,
                        help='Epochs per training phase (default: 10)')
    parser.add_argument('--gpus', type=str, default='0,1',
                        help='Comma-separated GPU IDs (default: 0,1)')
    parser.add_argument('--num-workers', type=int, default=None,
                        help='DataLoader workers (default: 4 on Windows, 16 on Linux)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--output-dir', type=str,
                        default=None,
                        help='Output directory (default: ./checkpoints/megastarid/arch_grid_search_YYYYMMDD_HHMMSS)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print experiment plan without running')
    
    # Grid options
    parser.add_argument('--architectures', type=str, nargs='+',
                        default=['swinv2-tiny', 'densenet121', 'densenet169', 'resnet50', 'convnext-tiny'],
                        choices=['swinv2-tiny', 'densenet121', 'densenet169', 'resnet50', 'convnext-tiny'],
                        help='Architectures to test')
    parser.add_argument('--strategies', type=str, nargs='+',
                        default=['star_only', 'pretrain_finetune', 'cotrain'],
                        choices=['star_only', 'pretrain_finetune', 'cotrain'],
                        help='Training strategies to test')
    parser.add_argument('--losses', type=str, nargs='+',
                        default=['triplet'],
                        choices=['circle', 'triplet'],
                        help='Loss functions to test (default: triplet only for arch comparison)')
    
    # Batch size control
    parser.add_argument('--max-batch-size', type=int, default=None,
                        help='Cap batch size (useful for small datasets like star_only)')
    
    # Negative-only identities experiment
    parser.add_argument('--test-negative-only', action='store_true',
                        help='Run experiments both with and without negative-only identities')
    parser.add_argument('--include-negative-only', action='store_true',
                        help='Include negative-only identities in training (single-outing stars)')
    
    # Run specific experiments
    parser.add_argument('--experiments', type=str, nargs='+', default=None,
                        help='Run only specific experiments (by name)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from previous run, skipping completed experiments')
    
    args = parser.parse_args()
    
    # Generate timestamped output directory if not specified
    if args.output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_dir = f'./checkpoints/megastarid/arch_grid_search_{timestamp}'
    
    # Parse GPU IDs
    gpu_ids = [int(x) for x in args.gpus.split(',')]
    num_gpus = len(gpu_ids)
    print(f"Using GPUs: {gpu_ids}")
    
    # Handle num_workers with Windows safety cap
    if args.num_workers is None:
        args.num_workers = MAX_WORKERS_WINDOWS if IS_WINDOWS else 16
    elif IS_WINDOWS and args.num_workers > MAX_WORKERS_WINDOWS:
        print(f"\n⚠️  Windows detected: Reducing num_workers from {args.num_workers} to {MAX_WORKERS_WINDOWS}")
        print(f"   (Windows multiprocessing can hang with >4 workers)")
        args.num_workers = MAX_WORKERS_WINDOWS
    
    print(f"DataLoader workers: {args.num_workers}")
    
    # Set seed
    set_seed(args.seed)
    
    # Determine negative-only options
    if args.test_negative_only:
        # Test both settings
        include_negative_only_options = [False, True]
    elif args.include_negative_only:
        # Only use include mode
        include_negative_only_options = [True]
    else:
        # Default: exclude negative-only
        include_negative_only_options = [False]
    
    # Create experiment grid
    experiments = create_experiment_grid(
        epochs=args.epochs,
        architectures=args.architectures,
        strategies=args.strategies,
        loss_types=args.losses,
        num_gpus=num_gpus,
        max_batch_size=args.max_batch_size,
        include_negative_only_options=include_negative_only_options,
    )
    
    # Filter if specific experiments requested
    if args.experiments:
        experiments = [e for e in experiments if e.name in args.experiments]
    
    # Resume: load existing results and skip completed experiments
    base_dir = Path(args.output_dir)
    existing_results = []
    completed_names = set()
    
    if args.resume:
        results_file = base_dir / 'results.json'
        if results_file.exists():
            with open(results_file, 'r') as f:
                data = json.load(f)
                existing_results = [
                    ArchExperimentResult(**r) for r in data.get('results', [])
                ]
                completed_names = {r.name for r in existing_results}
            
            # Filter out completed experiments
            original_count = len(experiments)
            experiments = [e for e in experiments if e.name not in completed_names]
            skipped = original_count - len(experiments)
            
            if skipped > 0:
                print(f"\n🔄 Resuming: skipping {skipped} completed experiments")
                print(f"   Completed: {', '.join(sorted(completed_names))}")
        else:
            print(f"\n⚠️  --resume specified but no results.json found at {results_file}")
            print("   Starting fresh...")
    
    # Print plan
    print("\n" + "=" * 80)
    print("ARCHITECTURE GRID SEARCH PLAN")
    print("=" * 80)
    print(f"Output directory: {args.output_dir}")
    print(f"Epochs per phase: {args.epochs}")
    print(f"GPUs: {gpu_ids} ({num_gpus} total)")
    print(f"Architectures: {args.architectures}")
    print(f"Strategies: {args.strategies}")
    print(f"Losses: {args.losses}")
    if args.test_negative_only:
        print(f"Negative-only IDs: Testing BOTH (with and without)")
    elif args.include_negative_only:
        print(f"Negative-only IDs: INCLUDED in training")
    else:
        print(f"Negative-only IDs: EXCLUDED from training (default)")
    print(f"\nExperiments to run ({len(experiments)}):")
    
    total_estimated_time = 0
    for i, exp in enumerate(experiments, 1):
        # Rough time estimate: ~2min per epoch for star_only, ~5min for others
        if exp.strategy == 'star_only':
            est_time = exp.finetune_epochs * 2
        elif exp.strategy == 'pretrain_finetune':
            est_time = (exp.pretrain_epochs + exp.finetune_epochs) * 5
        else:
            est_time = exp.cotrain_epochs * 6
        total_estimated_time += est_time
        
        neg_str = " [+neg]" if exp.include_negative_only else ""
        print(f"  {i:2}. {exp.name}")
        print(f"      Arch: {exp.architecture} | Batch: {exp.batch_size}{neg_str} | ~{est_time}min")
        print(f"      LR: {exp.learning_rate:.2e} | WD: {exp.weight_decay:.2e} | Warmup: {exp.warmup_ratio:.0%} | LLRD: {exp.use_llrd}")
    
    print(f"\nEstimated total time: ~{total_estimated_time/60:.1f} hours")
    
    if args.dry_run:
        print("\n[DRY RUN - not executing]")
        return
    
    # Show data paths being used
    from megastarid.config import FinetuneConfig
    test_config = FinetuneConfig()
    print(f"\n📂 Data paths:")
    print(f"   Wildlife10k: {test_config.wildlife_root}")
    print(f"   star_dataset: {test_config.star_dataset_root}")
    
    # Confirm
    print("\nStarting in 5 seconds... (Ctrl+C to cancel)")
    time.sleep(5)
    
    # Run experiments
    # base_dir already defined above for resume logic
    results = list(existing_results)  # Start with existing results if resuming
    
    for i, exp in enumerate(experiments, 1):
        print(f"\n\n{'#'*80}")
        print(f"# EXPERIMENT {i}/{len(experiments)}: {exp.name}")
        print(f"{'#'*80}")
        
        try:
            result = run_experiment(
                exp, gpu_ids, base_dir, args.seed, args.num_workers
            )
            results.append(result)
            
            # Save intermediate results
            save_results(results, base_dir / 'results.json')
            
            # Generate per-experiment training plots
            exp_dir = base_dir / exp.name
            
            # Find all training_history.json files (handles pretrain_finetune subdirs)
            history_files = list(exp_dir.glob('**/training_history.json'))
            for history_path in history_files:
                try:
                    # Determine experiment name from path
                    if history_path.parent == exp_dir:
                        plot_name = exp.name
                    else:
                        # Include subdirectory name (e.g., "pretrain" or "finetune")
                        plot_name = f"{exp.name}/{history_path.parent.name}"
                    
                    plot_training_history(history_path, experiment_name=plot_name)
                    print(f"📊 Training plot saved to: {history_path.parent / 'training_plots.png'}")
                except Exception as plot_err:
                    print(f"⚠️ Could not generate training plot for {history_path}: {plot_err}")
            
            # Clear memory between experiments
            clear_gpu_memory()
            
        except Exception as e:
            print(f"\n❌ Experiment {exp.name} failed: {e}")
            import traceback
            traceback.print_exc()
            clear_gpu_memory()
            continue
    
    # Final summary
    if results:
        print_summary(results)
        save_results(results, base_dir / 'results_final.json')
        
        # Generate grid search summary plot
        try:
            # Convert ArchExperimentResult to ExperimentResult for plotting
            plot_results = [
                ExperimentResult(
                    name=r.name,
                    architecture=r.architecture,
                    strategy=r.strategy,
                    loss_type=r.loss_type,
                    total_params_millions=r.total_params_millions,
                    batch_size=r.batch_size,
                    star_mAP=r.star_mAP,
                    star_rank1=r.star_rank1,
                    star_rank5=r.star_rank5,
                    total_time_seconds=r.total_time_seconds,
                    best_epoch=r.best_epoch,
                )
                for r in results
            ]
            plot_grid_search_summary(plot_results, base_dir)
        except Exception as plot_err:
            print(f"⚠️ Could not generate grid search summary plot: {plot_err}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()

