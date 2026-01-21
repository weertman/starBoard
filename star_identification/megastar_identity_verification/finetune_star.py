#!/usr/bin/env python
"""
Fine-tune verification model on star_dataset.

Supports two modes:
1. From verification checkpoint: --pretrain-checkpoint (full verification model)
2. From backbone checkpoint: --backbone-checkpoint (embedding model, creates fresh cross-attention)

Usage:
    # From verification model
    python -m megastar_identity_verification.finetune_star \
        --pretrain-checkpoint checkpoints/verification/pretrain_.../best.pth \
        --epochs 50 --batch-size 16
    
    # From backbone (embedding) model
    python -m megastar_identity_verification.finetune_star \
        --backbone-checkpoint checkpoints/megastarid/.../best.pth \
        --epochs 50 --batch-size 16
"""
import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from megastar_identity_verification.config import (
    VerificationConfig, BackboneConfig, CrossAttentionConfig, FinetuneConfig
)
from megastar_identity_verification.model import VerificationModel, create_verification_model
from megastar_identity_verification.dataset import StarDatasetPairDataset, create_pair_dataloaders
from megastar_identity_verification.transforms import get_train_transforms, get_test_transforms
from megastar_identity_verification.trainer import VerificationTrainer


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description='Fine-tune verification model on star_dataset')
    
    # Model - two mutually exclusive options
    checkpoint_group = parser.add_mutually_exclusive_group(required=True)
    checkpoint_group.add_argument('--pretrain-checkpoint', type=str,
                        help='Path to pre-trained verification model checkpoint')
    checkpoint_group.add_argument('--backbone-checkpoint', type=str,
                        help='Path to backbone (embedding) model checkpoint')
    parser.add_argument('--freeze-backbone', action='store_true',
                        help='Freeze backbone weights during fine-tuning')
    
    # Training
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs (default: 50)')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size (pairs per batch, default: 16)')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='Learning rate (default: 1e-5, lower for fine-tuning)')
    parser.add_argument('--pairs-per-epoch', type=int, default=10000,
                        help='Number of pairs to sample per epoch (default: 10000)')
    parser.add_argument('--val-every', type=int, default=5,
                        help='Validate every N epochs (default: 5)')
    
    # Data
    parser.add_argument('--star-root', type=str, default='./star_dataset_resized',
                        help='Path to star_dataset')
    parser.add_argument('--include-inaturalist', action='store_true',
                        help='Include iNaturalist data (excluded by default)')
    
    # Hardware
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU index to use (default: 0)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='DataLoader workers (default: 4)')
    parser.add_argument('--no-amp', action='store_true',
                        help='Disable automatic mixed precision')
    
    # Output
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: auto-generated)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"checkpoints/verification/finetune_star_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine checkpoint type
    checkpoint_path = args.pretrain_checkpoint or args.backbone_checkpoint
    checkpoint_type = "verification" if args.pretrain_checkpoint else "backbone"
    
    print(f"\n{'='*60}")
    print("VERIFICATION MODEL FINE-TUNING ON STAR_DATASET")
    print(f"{'='*60}")
    print(f"Checkpoint ({checkpoint_type}): {checkpoint_path}")
    print(f"Freeze backbone: {args.freeze_backbone}")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.lr}")
    print(f"Pairs per epoch: {args.pairs_per_epoch}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")
    
    # Device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        torch.cuda.set_device(args.gpu)
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Model config
    model_config = VerificationConfig(
        backbone=BackboneConfig(
            name="convnext-small",
            freeze=args.freeze_backbone,
        ),
        cross_attention=CrossAttentionConfig(
            feature_dim=768,
            hidden_dim=256,
            num_layers=2,
            num_heads=8,
        ),
    )
    
    if args.pretrain_checkpoint:
        # Load full verification model from checkpoint
        print("\nLoading pre-trained verification model...")
        checkpoint = torch.load(args.pretrain_checkpoint, map_location='cpu', weights_only=False)
        model = VerificationModel(model_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded verification model from epoch {checkpoint['epoch']}")
    else:
        # Create fresh verification model and load backbone from embedding checkpoint
        print("\nCreating verification model with backbone from embedding checkpoint...")
        model = create_verification_model(
            config=model_config,
            backbone_checkpoint=args.backbone_checkpoint,
        )
    
    # Optionally freeze backbone
    if args.freeze_backbone:
        for param in model.backbone.parameters():
            param.requires_grad = False
        print("Backbone frozen")
    
    # Create datasets
    print("\nLoading star_dataset...")
    train_transform = get_train_transforms(224)
    test_transform = get_test_transforms(224)
    
    train_dataset = StarDatasetPairDataset.from_star_dataset(
        data_root=args.star_root,
        mode='train',
        transform=train_transform,
        pairs_per_epoch=args.pairs_per_epoch,
        positive_ratio=0.5,
        seed=args.seed,
        include_inaturalist=args.include_inaturalist,
    )
    
    val_dataset = StarDatasetPairDataset.from_star_dataset(
        data_root=args.star_root,
        mode='test',
        transform=test_transform,
        pairs_per_epoch=2000,  # Smaller validation set
        positive_ratio=0.5,
        seed=args.seed + 1,
        include_inaturalist=args.include_inaturalist,
    )
    
    # Create dataloaders
    train_loader = create_pair_dataloaders(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    val_loader = create_pair_dataloaders(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Create training config
    train_config = FinetuneConfig(
        checkpoint_dir=str(output_dir),
        star_dataset_root=args.star_root,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        pairs_per_epoch=args.pairs_per_epoch,
        num_workers=args.num_workers,
        use_amp=not args.no_amp,
        seed=args.seed,
    )
    
    # Create trainer
    trainer = VerificationTrainer(model, train_config, device)
    
    # Train
    best_metrics = trainer.train(
        train_loader,
        val_loader,
        num_epochs=args.epochs,
        val_every=args.val_every,
    )
    
    # Save experiment summary
    summary = {
        'checkpoint_type': checkpoint_type,
        'checkpoint_path': checkpoint_path,
        'freeze_backbone': args.freeze_backbone,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'pairs_per_epoch': args.pairs_per_epoch,
        'learning_rate': args.lr,
        'best_metrics': best_metrics,
    }
    
    with open(output_dir / 'experiment_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n[DONE] Fine-tuning complete! Results saved to: {output_dir}")


if __name__ == '__main__':
    main()

