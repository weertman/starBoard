#!/usr/bin/env python
"""
Pre-training on Wildlife10k dataset.

Trains the model on the full Wildlife10k dataset (excluding SeaStarReID2023)
to learn general visual features for animal re-identification.

Usage:
    python -m megastarid.pretrain --epochs 50
    python -m megastarid.pretrain --epochs 50 --batch-size 32
    python -m megastarid.pretrain --exclude-datasets SeaStarReID2023 SMALST
"""
import argparse
import random
import numpy as np
import torch
from pathlib import Path

from .config import PretrainConfig, ModelConfig, LossConfig
from .datasets import create_pretrain_dataloaders
from .models import create_model, count_parameters
from .trainer import MegaStarTrainer


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description='Pre-train on Wildlife10k')
    
    # Data
    parser.add_argument('--wildlife-root', type=str, default='./wildlifeReID',
                        help='Path to wildlifeReID folder')
    parser.add_argument('--include-datasets', type=str, nargs='+', default=[],
                        help='Only include these datasets (empty = all)')
    parser.add_argument('--exclude-datasets', type=str, nargs='+', 
                        default=['SeaStarReID2023'],
                        help='Exclude these datasets')
    parser.add_argument('--split-strategy', type=str, default='original',
                        choices=['original', 'recommended', 'random'],
                        help='Split strategy')
    
    # Model
    parser.add_argument('--model', type=str,
                        default='microsoft/swinv2-small-patch4-window16-256',
                        help='Model name')
    parser.add_argument('--embedding-dim', type=int, default=512,
                        help='Embedding dimension')
    parser.add_argument('--image-size', type=int, default=384,
                        help='Image size')
    
    # Training
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--num-instances', type=int, default=4,
                        help='Instances per identity (K in P-K)')
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
                        default='./checkpoints/megastarid/pretrain',
                        help='Checkpoint directory')
    parser.add_argument('--val-every', type=int, default=5,
                        help='Validate every N epochs')
    
    # Other
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Build config
    config = PretrainConfig(
        wildlife_root=args.wildlife_root,
        checkpoint_dir=args.checkpoint_dir,
        model=ModelConfig(
            name=args.model,
            embedding_dim=args.embedding_dim,
            image_size=args.image_size,
        ),
        loss=LossConfig(
            circle_weight=args.circle_weight,
            triplet_weight=args.triplet_weight,
        ),
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        num_instances=args.num_instances,
        learning_rate=args.lr,
        warmup_epochs=args.warmup_epochs,
        device=args.device,
        num_workers=args.num_workers,
        use_amp=not args.no_amp,
        seed=args.seed,
        include_datasets=args.include_datasets,
        exclude_datasets=args.exclude_datasets,
        split_strategy=args.split_strategy,
        val_every_n_epochs=args.val_every,
    )
    
    # Print header
    print("=" * 60)
    print("MEGASTARID PRE-TRAINING")
    print("=" * 60)
    print(f"\nWildlife10k root: {args.wildlife_root}")
    print(f"Excluding: {args.exclude_datasets}")
    print(f"Device: {args.device}")
    
    # Load data
    print("\n" + "-" * 60)
    print("LOADING DATA")
    print("-" * 60)
    
    train_loader, test_loader = create_pretrain_dataloaders(config)
    
    # Create model
    print("\n" + "-" * 60)
    print("MODEL")
    print("-" * 60)
    
    model = create_model(config)
    params = count_parameters(model)
    
    print(f"Model: {config.model.name.split('/')[-1]}")
    print(f"Image size: {config.model.image_size}px")
    print(f"Embedding dim: {config.model.embedding_dim}")
    print(f"Parameters: {params['total']/1e6:.1f}M total, {params['trainable']/1e6:.1f}M trainable")
    
    # Create trainer
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    trainer = MegaStarTrainer(
        model=model,
        config=config,
        device=device,
        mode='pretrain',
    )
    
    # Train
    best_metrics = trainer.train_pretrain(
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=args.epochs,
        val_every=args.val_every,
    )
    
    # Save config
    config.save(Path(args.checkpoint_dir) / 'config.json')
    
    print("\n" + "=" * 60)
    print("PRE-TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best metrics:")
    for k, v in best_metrics.items():
        print(f"  {k}: {v:.4f}")
    print(f"\nCheckpoints saved to: {args.checkpoint_dir}")
    print(f"\nNext step: Fine-tune on star_dataset:")
    print(f"  python -m megastarid.finetune --checkpoint {args.checkpoint_dir}/best.pth")


if __name__ == '__main__':
    main()


