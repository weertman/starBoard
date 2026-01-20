#!/usr/bin/env python
"""
Training script for Wildlife ReID-10k.

Uses the same model architecture as temporal_reid (SwinV2 + GeM + embedding head).

Usage:
    python -m wildlife_reid.train
    python -m wildlife_reid.train --epochs 50 --batch-size 32
    python -m wildlife_reid.train --include-datasets SeaStarReID2023 BelugaID
"""
import argparse
import random
import numpy as np
import torch
from pathlib import Path

from .config import Wildlife10kConfig, FilterConfig, SplitConfig
from .loader import Wildlife10kLoader
from .models import create_model
from .training import Trainer
from .utils.transforms import get_train_transforms, get_test_transforms


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model) -> tuple:
    """Count model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main():
    parser = argparse.ArgumentParser(description='Train on Wildlife ReID-10k')
    
    # Data
    parser.add_argument('--data-root', type=str, default='./wildlifeReID',
                        help='Path to wildlifeReID folder')
    
    # Filtering
    parser.add_argument('--include-datasets', type=str, nargs='+', default=None,
                        help='Only include these datasets')
    parser.add_argument('--exclude-datasets', type=str, nargs='+', default=None,
                        help='Exclude these datasets')
    parser.add_argument('--include-species', type=str, nargs='+', default=None,
                        help='Only include these species')
    
    # Split strategy
    parser.add_argument('--split-strategy', type=str, default='original',
                        choices=['original', 'recommended', 'time_aware', 'cluster_aware', 'random'],
                        help='Split strategy')
    
    # Model
    parser.add_argument('--model', type=str, 
                        default='microsoft/swinv2-small-patch4-window16-256',
                        help='Model name')
    parser.add_argument('--embedding-dim', type=int, default=512,
                        help='Embedding dimension')
    parser.add_argument('--image-size', type=int, default=384,
                        help='Input image size')
    
    # Training
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--num-instances', type=int, default=4,
                        help='Instances per identity (K in P-K sampling)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--warmup-epochs', type=int, default=5,
                        help='Warmup epochs')
    
    # Loss
    parser.add_argument('--circle-weight', type=float, default=0.7,
                        help='Circle loss weight')
    parser.add_argument('--triplet-weight', type=float, default=0.3,
                        help='Triplet loss weight')
    
    # Hardware
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='DataLoader workers')
    parser.add_argument('--no-amp', action='store_true',
                        help='Disable mixed precision')
    
    # Output
    parser.add_argument('--checkpoint-dir', type=str, 
                        default='./checkpoints/wildlife10k',
                        help='Checkpoint directory')
    parser.add_argument('--val-every', type=int, default=5,
                        help='Validate every N epochs')
    
    # Other
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Build config
    filter_config = FilterConfig(
        include_datasets=args.include_datasets or [],
        exclude_datasets=args.exclude_datasets or [],
        include_species=args.include_species or [],
    )
    
    split_config = SplitConfig(
        strategy=args.split_strategy,
        seed=args.seed,
    )
    
    config = Wildlife10kConfig(
        data_root=args.data_root,
        filter=filter_config,
        split=split_config,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_instances=args.num_instances,
        model_name=args.model,
        embedding_dim=args.embedding_dim,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_epochs=args.warmup_epochs,
        circle_weight=args.circle_weight,
        triplet_weight=args.triplet_weight,
        device=args.device,
        num_workers=args.num_workers,
        use_amp=not args.no_amp,
        checkpoint_dir=args.checkpoint_dir,
        seed=args.seed,
    )
    
    # Print header
    print("=" * 60)
    print("WILDLIFE REID-10K TRAINING")
    print("=" * 60)
    print(f"\nConfig: {config.summary()}")
    print(f"Device: {args.device}")
    
    # Load data
    print("\n" + "-" * 60)
    print("LOADING DATA")
    print("-" * 60)
    
    train_transform = get_train_transforms(config.image_size)
    test_transform = get_test_transforms(config.image_size)
    
    loader = Wildlife10kLoader(args.data_root, config)
    train_dataset, test_dataset = loader.load(train_transform, test_transform)
    train_loader, test_loader = loader.create_dataloaders()
    
    train_dataset.print_summary()
    test_dataset.print_summary()
    
    # Create model
    print("\n" + "-" * 60)
    print("MODEL")
    print("-" * 60)
    
    model = create_model(config)
    total_params, trainable_params = count_parameters(model)
    
    print(f"Model: {config.model_name.split('/')[-1]}")
    print(f"Image size: {config.image_size}px")
    print(f"Embedding dim: {config.embedding_dim}")
    print(f"Parameters: {total_params / 1e6:.1f}M total, {trainable_params / 1e6:.1f}M trainable")
    
    # Create trainer
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    trainer = Trainer(
        model=model,
        config=config,
        device=device,
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
    
    best_metrics = trainer.train(
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=args.epochs,
        val_every_n_epochs=args.val_every,
    )
    
    # Save final config
    config.save(Path(args.checkpoint_dir) / 'config.json')
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best metrics:")
    for k, v in best_metrics.items():
        print(f"  {k}: {v:.4f}")
    print(f"\nCheckpoints saved to: {args.checkpoint_dir}")


if __name__ == '__main__':
    main()


