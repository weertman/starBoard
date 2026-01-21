#!/usr/bin/env python
"""
Main training script for temporal re-identification.

Usage:
    python -m temporal_reid.train --dataset-root ./star_dataset
    python -m temporal_reid.train --config config.json
"""
import argparse
import sys
from pathlib import Path

import torch

from .config import Config
from .data import (
    prepare_temporal_split, create_dataloaders, create_train_transform, create_val_transform,
    create_cached_eval_loaders
)
from .models import create_model
from .training import Trainer
from .utils import set_seed, get_device, count_parameters


def main():
    parser = argparse.ArgumentParser(description='Train temporal re-identification model')
    
    # Data
    parser.add_argument('--dataset-root', type=str, default='./star_dataset',
                        help='Path to star dataset')
    parser.add_argument('--metadata', type=str, default=None,
                        help='Path to existing metadata CSV (skip preparation)')
    
    # Config
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config JSON file')
    
    # Overrides
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate')
    parser.add_argument('--embedding-dim', type=int, default=None,
                        help='Embedding dimension')
    parser.add_argument('--model', type=str, default=None,
                        help='Model name/path (e.g., microsoft/swinv2-small-patch4-window16-256)')
    parser.add_argument('--image-size', type=int, default=None,
                        help='Input image size (overrides model default, e.g., 384)')
    
    # Training
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints/temporal',
                        help='Checkpoint directory')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    
    # Other
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--force-prepare', action='store_true',
                        help='Force regeneration of metadata')
    parser.add_argument('--gpus', type=str, default=None,
                        help='GPU IDs to use (e.g., "0,1" for DataParallel). Default: single GPU.')
    parser.add_argument('--cache-eval', action='store_true',
                        help='Cache evaluation images in RAM for faster validation')
    parser.add_argument('--eval-batch-size', type=int, default=None,
                        help='Batch size for evaluation (default: 4x training batch size)')
    parser.add_argument('--cache-workers', type=int, default=16,
                        help='Number of workers for background image caching')
    
    args = parser.parse_args()
    
    # Load or create config
    if args.config:
        config = Config.load(args.config)
    else:
        config = Config()
    
    # Apply overrides
    config.star_dataset_root = args.dataset_root
    config.checkpoint_dir = args.checkpoint_dir
    config.seed = args.seed
    
    if args.epochs:
        config.num_epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.learning_rate = args.lr
    if args.embedding_dim:
        config.embedding_dim = args.embedding_dim
    if args.model:
        config.model_name = args.model
    if args.image_size:
        config.image_size = args.image_size
    
    # Set seed
    set_seed(config.seed)
    
    # Get device and parse GPU IDs
    device = get_device(config.device)
    
    gpu_ids = None
    if args.gpus:
        gpu_ids = [int(g.strip()) for g in args.gpus.split(",")]
        # Set primary device to first GPU
        device = torch.device(f"cuda:{gpu_ids[0]}")
    
    print("=" * 60)
    print("TEMPORAL RE-IDENTIFICATION TRAINING")
    print("=" * 60)
    print(f"\nConfig: {config.summary()}")
    if gpu_ids and len(gpu_ids) > 1:
        print(f"Device: {device} (DataParallel on GPUs {gpu_ids})")
    else:
        print(f"Device: {device}")
    
    # Prepare data
    print("\n" + "-" * 60)
    print("DATA PREPARATION")
    print("-" * 60)
    
    if args.metadata:
        from .data import load_metadata
        df = load_metadata(args.metadata)
    else:
        df = prepare_temporal_split(
            dataset_root=config.star_dataset_root,
            train_outing_ratio=config.temporal_split.train_outing_ratio,
            seed=config.temporal_split.seed,
            force_regenerate=args.force_prepare,
        )
    
    # Create transforms
    train_transform = create_train_transform(config)
    val_transform = create_val_transform(config)
    
    # Create dataloaders
    print("\n" + "-" * 60)
    print("CREATING DATALOADERS")
    print("-" * 60)
    
    train_loader, gallery_loader, query_loader = create_dataloaders(
        df, config, train_transform, val_transform
    )
    
    # Optionally create cached evaluation loaders for faster validation
    cacher = None
    if args.cache_eval:
        print("\n" + "-" * 60)
        print("SETTING UP RAM-CACHED EVALUATION")
        print("-" * 60)
        
        eval_batch_size = args.eval_batch_size or config.batch_size * 4
        
        # Get identity mapping from train dataset
        train_dataset = train_loader.dataset
        identity_to_label = train_dataset.identity_to_label
        
        gallery_loader, query_loader, cacher = create_cached_eval_loaders(
            df=df,
            val_transform=val_transform,
            identity_to_label=identity_to_label,
            batch_size=eval_batch_size,
            num_workers=args.cache_workers,
            pin_memory=config.pin_memory,
        )
        
        print(f"Eval batch size: {eval_batch_size} (vs training: {config.batch_size})")
        print(f"Cache workers: {args.cache_workers}")
        print("Background caching will start with training...")
    
    # Get number of classes (all identities, for loss function)
    num_classes = df['identity'].nunique()
    
    # Create model
    print("\n" + "-" * 60)
    print("MODEL")
    print("-" * 60)
    
    model = create_model(config)
    total_params, trainable_params = count_parameters(model)
    
    print(f"Model: {config.model_name.split('/')[-1]}")
    print(f"Image size: {config.get_image_size()}px")
    print(f"Embedding dim: {config.embedding_dim}")
    print(f"Parameters: {total_params / 1e6:.1f}M total, {trainable_params / 1e6:.1f}M trainable")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        config=config,
        num_classes=num_classes,
        device=device,
        gpu_ids=gpu_ids,
    )
    
    # Resume if specified
    start_epoch = 0
    if args.resume:
        print(f"\nResuming from {args.resume}")
        start_epoch, _ = trainer.load_checkpoint(args.resume)
    
    # Train
    print("\n" + "-" * 60)
    print("TRAINING")
    print("-" * 60)
    
    # Start background caching if enabled
    if cacher is not None:
        cacher.start()
    
    best_metrics = trainer.train(
        train_loader=train_loader,
        query_loader=query_loader,
        gallery_loader=gallery_loader,
        eval_cacher=cacher,
    )
    
    # Save final config
    config.save(Path(config.checkpoint_dir) / 'config.json')
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best metrics:")
    for k, v in best_metrics.items():
        print(f"  {k}: {v:.4f}")
    print(f"\nCheckpoints saved to: {config.checkpoint_dir}")


if __name__ == '__main__':
    main()

