#!/usr/bin/env python
"""
Pre-train verification model on Wildlife10k.

Uses pairs from Wildlife10k to train the cross-attention verification model.
Backbone is loaded from a pre-trained embedding model checkpoint.

Usage:
    python -m megastar_identity_verification.pretrain_wildlife \
        --backbone-checkpoint checkpoints/megastarid/.../best.pth \
        --epochs 100 \
        --batch-size 32
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
    VerificationConfig, BackboneConfig, CrossAttentionConfig, PretrainConfig
)
from megastar_identity_verification.model import create_verification_model
from megastar_identity_verification.dataset import Wildlife10kPairDataset, create_pair_dataloaders
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
    parser = argparse.ArgumentParser(description='Pre-train verification model on Wildlife10k')
    
    # Model
    parser.add_argument('--backbone-checkpoint', type=str, required=True,
                        help='Path to embedding model checkpoint for backbone weights')
    parser.add_argument('--freeze-backbone', action='store_true',
                        help='Freeze backbone weights (only train cross-attention)')
    parser.add_argument('--hidden-dim', type=int, default=256,
                        help='Hidden dimension for cross-attention (default: 256)')
    parser.add_argument('--num-layers', type=int, default=2,
                        help='Number of cross-attention layers (default: 2)')
    parser.add_argument('--num-heads', type=int, default=8,
                        help='Number of attention heads (default: 8)')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size (pairs per batch, default: 32)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--pairs-per-epoch', type=int, default=50000,
                        help='Number of pairs to sample per epoch (default: 50000)')
    parser.add_argument('--val-every', type=int, default=5,
                        help='Validate every N epochs (default: 5)')
    
    # Data
    parser.add_argument('--wildlife-root', type=str, default='./wildlifeReID_resized',
                        help='Path to Wildlife10k data')
    parser.add_argument('--split-strategy', type=str, default='recommended',
                        help='Split strategy for Wildlife10k')
    
    # Hardware
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU index to use (default: 0)')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='DataLoader workers (default: 8)')
    parser.add_argument('--no-amp', action='store_true',
                        help='Disable automatic mixed precision')
    
    # Output
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: auto-generated)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    
    # Resume
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"checkpoints/verification/pretrain_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("VERIFICATION MODEL PRE-TRAINING ON WILDLIFE10K")
    print(f"{'='*60}")
    print(f"Backbone checkpoint: {args.backbone_checkpoint}")
    print(f"Freeze backbone: {args.freeze_backbone}")
    print(f"Cross-attention: {args.num_layers} layers, {args.hidden_dim} dim, {args.num_heads} heads")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}")
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
    
    # Create model config
    backbone_config = BackboneConfig(
        name="convnext-small",
        checkpoint_path=args.backbone_checkpoint,
        freeze=args.freeze_backbone,
    )
    
    cross_attn_config = CrossAttentionConfig(
        feature_dim=768,  # ConvNeXt-small feature dim
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
    )
    
    model_config = VerificationConfig(
        backbone=backbone_config,
        cross_attention=cross_attn_config,
    )
    
    # Create model
    print("\nCreating verification model...")
    model = create_verification_model(
        config=model_config,
        backbone_checkpoint=args.backbone_checkpoint,
    )
    
    # Create datasets
    print("\nLoading Wildlife10k datasets...")
    train_transform = get_train_transforms(224)
    test_transform = get_test_transforms(224)
    
    train_dataset = Wildlife10kPairDataset.from_wildlife10k(
        data_root=args.wildlife_root,
        mode='train',
        split_strategy=args.split_strategy,
        exclude_datasets=["SeaStarReID2023"],
        transform=train_transform,
        pairs_per_epoch=args.pairs_per_epoch,
        positive_ratio=0.5,
        seed=args.seed,
    )
    
    val_dataset = Wildlife10kPairDataset.from_wildlife10k(
        data_root=args.wildlife_root,
        mode='test',
        split_strategy=args.split_strategy,
        exclude_datasets=["SeaStarReID2023"],
        transform=test_transform,
        pairs_per_epoch=10000,  # Fewer pairs for validation
        positive_ratio=0.5,
        seed=args.seed + 1,  # Different seed for val
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
    train_config = PretrainConfig(
        checkpoint_dir=str(output_dir),
        wildlife_root=args.wildlife_root,
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
    
    # Resume if specified
    if args.resume:
        print(f"\nResuming from: {args.resume}")
        start_epoch = trainer.load_checkpoint(args.resume)
        print(f"Resumed from epoch {start_epoch}")
    
    # Train
    best_metrics = trainer.train(
        train_loader,
        val_loader,
        num_epochs=args.epochs,
        val_every=args.val_every,
    )
    
    # Save experiment summary
    summary = {
        'backbone_checkpoint': args.backbone_checkpoint,
        'freeze_backbone': args.freeze_backbone,
        'cross_attention': {
            'hidden_dim': args.hidden_dim,
            'num_layers': args.num_layers,
            'num_heads': args.num_heads,
        },
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'pairs_per_epoch': args.pairs_per_epoch,
        'learning_rate': args.lr,
        'best_metrics': best_metrics,
    }
    
    with open(output_dir / 'experiment_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n[DONE] Experiment complete! Results saved to: {output_dir}")


if __name__ == '__main__':
    main()

