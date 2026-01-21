#!/usr/bin/env python
"""
Co-training on Wildlife10k + star_dataset simultaneously.

Trains on mixed batches from both datasets to learn general
animal features while specializing on sea stars.

Usage:
    python -m megastarid.cotrain --epochs 100
    python -m megastarid.cotrain --epochs 100 --star-batch-ratio 0.3
    python -m megastarid.cotrain --epochs 100 --alternate-batches
"""
import argparse
import random
import numpy as np
import torch
from pathlib import Path

from .config import CotrainConfig, ModelConfig, LossConfig
from .datasets import create_cotrain_dataloaders
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
    parser = argparse.ArgumentParser(description='Co-train on Wildlife10k + star_dataset')
    
    # Data paths
    parser.add_argument('--wildlife-root', type=str, default='./wildlifeReID',
                        help='Path to wildlifeReID folder')
    parser.add_argument('--star-dataset-root', type=str, default='./star_dataset',
                        help='Path to star_dataset folder')
    
    # Wildlife filtering
    parser.add_argument('--include-datasets', type=str, nargs='+', default=[],
                        help='Only include these wildlife datasets')
    parser.add_argument('--exclude-datasets', type=str, nargs='+',
                        default=['SeaStarReID2023'],
                        help='Exclude these wildlife datasets')
    parser.add_argument('--split-strategy', type=str, default='original',
                        choices=['original', 'recommended', 'random'],
                        help='Wildlife split strategy')
    
    # Star dataset
    parser.add_argument('--train-outing-ratio', type=float, default=0.8,
                        help='Star dataset train outing ratio')
    parser.add_argument('--min-outings', type=int, default=2,
                        help='Min outings for evaluable identity')
    
    # Batch composition
    parser.add_argument('--star-batch-ratio', type=float, default=0.3,
                        help='Fraction of batch from star_dataset (0.3 = 30% stars)')
    parser.add_argument('--alternate-batches', action='store_true',
                        help='Alternate full batches between datasets')
    
    # Model
    parser.add_argument('--model', type=str,
                        default='microsoft/swinv2-small-patch4-window16-256',
                        help='Model name')
    parser.add_argument('--embedding-dim', type=int, default=512,
                        help='Embedding dimension')
    parser.add_argument('--image-size', type=int, default=384,
                        help='Image size')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100,
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
                        default='./checkpoints/megastarid/cotrain',
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
    config = CotrainConfig(
        wildlife_root=args.wildlife_root,
        star_dataset_root=args.star_dataset_root,
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
        train_outing_ratio=args.train_outing_ratio,
        min_outings_for_eval=args.min_outings,
        star_batch_ratio=args.star_batch_ratio,
        alternate_batches=args.alternate_batches,
        val_every_n_epochs=args.val_every,
    )
    
    # Print header
    print("=" * 60)
    print("MEGASTARID CO-TRAINING")
    print("=" * 60)
    print(f"\nWildlife10k: {args.wildlife_root}")
    print(f"Star dataset: {args.star_dataset_root}")
    print(f"Excluding: {args.exclude_datasets}")
    if args.alternate_batches:
        print(f"Mode: Alternating batches")
    else:
        print(f"Mode: Mixed batches ({int(args.star_batch_ratio*100)}% stars)")
    print(f"Device: {args.device}")
    
    # Load data
    print("\n" + "-" * 60)
    print("LOADING DATA")
    print("-" * 60)
    
    train_loader, wildlife_test_loader, star_gallery_loader, star_query_loader = \
        create_cotrain_dataloaders(config)
    
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
        mode='cotrain',
    )
    
    # Train
    best_metrics = trainer.train_cotrain(
        train_loader=train_loader,
        wildlife_test_loader=wildlife_test_loader,
        star_gallery_loader=star_gallery_loader,
        star_query_loader=star_query_loader,
        num_epochs=args.epochs,
        val_every=args.val_every,
    )
    
    # Save config
    config.save(Path(args.checkpoint_dir) / 'config.json')
    
    print("\n" + "=" * 60)
    print("CO-TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best star_dataset metrics:")
    for k, v in best_metrics.items():
        print(f"  {k}: {v:.4f}")
    print(f"\nCheckpoints saved to: {args.checkpoint_dir}")


if __name__ == '__main__':
    main()


