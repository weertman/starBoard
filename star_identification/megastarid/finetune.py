#!/usr/bin/env python
"""
Fine-tuning on star_dataset.

Loads a pre-trained checkpoint and fine-tunes on the star_dataset
for sea star re-identification.

Usage:
    python -m megastarid.finetune --checkpoint checkpoints/megastarid/pretrain/best.pth
    python -m megastarid.finetune --checkpoint checkpoints/megastarid/pretrain/best.pth --epochs 100
    python -m megastarid.finetune --epochs 50  # No pre-training (baseline)
"""
import argparse
import random
import numpy as np
import torch
from pathlib import Path

from .config import FinetuneConfig, ModelConfig, LossConfig
from .datasets import create_finetune_dataloaders
from .models import create_model, load_pretrained_model, count_parameters, freeze_backbone, unfreeze_backbone
from .trainer import MegaStarTrainer


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description='Fine-tune on star_dataset')
    
    # Data
    parser.add_argument('--star-dataset-root', type=str, default='./star_dataset',
                        help='Path to star_dataset folder')
    parser.add_argument('--train-outing-ratio', type=float, default=0.8,
                        help='Fraction of outings for training')
    parser.add_argument('--min-outings', type=int, default=2,
                        help='Min outings for evaluable identity')
    
    # Pre-trained checkpoint
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to pre-trained checkpoint (None = train from scratch)')
    parser.add_argument('--freeze-backbone-epochs', type=int, default=0,
                        help='Freeze backbone for N epochs (0 = no freezing)')
    
    # Model (used if no checkpoint)
    parser.add_argument('--model', type=str,
                        default='microsoft/swinv2-small-patch4-window16-256',
                        help='Model name (ignored if --checkpoint provided)')
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
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='Learning rate (lower for fine-tuning)')
    parser.add_argument('--warmup-epochs', type=int, default=3,
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
                        default='./checkpoints/megastarid/finetune',
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
    config = FinetuneConfig(
        star_dataset_root=args.star_dataset_root,
        checkpoint_dir=args.checkpoint_dir,
        pretrain_checkpoint=args.checkpoint,
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
        freeze_backbone_epochs=args.freeze_backbone_epochs,
        device=args.device,
        num_workers=args.num_workers,
        use_amp=not args.no_amp,
        seed=args.seed,
        train_outing_ratio=args.train_outing_ratio,
        min_outings_for_eval=args.min_outings,
        val_every_n_epochs=args.val_every,
    )
    
    # Print header
    print("=" * 60)
    print("MEGASTARID FINE-TUNING")
    print("=" * 60)
    print(f"\nStar dataset: {args.star_dataset_root}")
    if args.checkpoint:
        print(f"Pre-trained checkpoint: {args.checkpoint}")
    else:
        print("Training from scratch (no pre-training)")
    print(f"Device: {args.device}")
    
    # Load data
    print("\n" + "-" * 60)
    print("LOADING DATA")
    print("-" * 60)
    
    train_loader, gallery_loader, query_loader = create_finetune_dataloaders(config)
    
    # Create/load model
    print("\n" + "-" * 60)
    print("MODEL")
    print("-" * 60)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    if args.checkpoint:
        print(f"Loading pre-trained weights from: {args.checkpoint}")
        model = load_pretrained_model(config, args.checkpoint, device)
    else:
        print("Creating fresh model (no pre-training)")
        model = create_model(config)
    
    params = count_parameters(model)
    print(f"Model: {config.model.name.split('/')[-1]}")
    print(f"Parameters: {params['total']/1e6:.1f}M total, {params['trainable']/1e6:.1f}M trainable")
    
    # Optional backbone freezing
    if args.freeze_backbone_epochs > 0:
        print(f"\nFreezing backbone for first {args.freeze_backbone_epochs} epochs")
        freeze_backbone(model)
        params = count_parameters(model)
        print(f"After freezing: {params['trainable']/1e6:.1f}M trainable")
    
    # Create trainer
    trainer = MegaStarTrainer(
        model=model,
        config=config,
        device=device,
        mode='finetune',
    )
    
    # Train with optional unfreezing
    if args.freeze_backbone_epochs > 0:
        # Phase 1: Frozen backbone
        print(f"\n--- Phase 1: Backbone frozen ({args.freeze_backbone_epochs} epochs) ---")
        trainer.train_finetune(
            train_loader=train_loader,
            gallery_loader=gallery_loader,
            query_loader=query_loader,
            num_epochs=args.freeze_backbone_epochs,
            val_every=args.val_every,
        )
        
        # Phase 2: Unfreeze and continue
        print(f"\n--- Phase 2: Full fine-tuning ({args.epochs - args.freeze_backbone_epochs} epochs) ---")
        unfreeze_backbone(model)
        
        # Reset optimizer with lower LR for fine-tuning
        trainer.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr / 10,  # Lower LR for fine-tuning after warmup
            weight_decay=config.weight_decay,
        )
        
        best_metrics = trainer.train_finetune(
            train_loader=train_loader,
            gallery_loader=gallery_loader,
            query_loader=query_loader,
            num_epochs=args.epochs - args.freeze_backbone_epochs,
            val_every=args.val_every,
        )
    else:
        # Standard fine-tuning
        best_metrics = trainer.train_finetune(
            train_loader=train_loader,
            gallery_loader=gallery_loader,
            query_loader=query_loader,
            num_epochs=args.epochs,
            val_every=args.val_every,
        )
    
    # Save config
    config.save(Path(args.checkpoint_dir) / 'config.json')
    
    print("\n" + "=" * 60)
    print("FINE-TUNING COMPLETE")
    print("=" * 60)
    print(f"Best metrics:")
    for k, v in best_metrics.items():
        print(f"  {k}: {v:.4f}")
    print(f"\nCheckpoints saved to: {args.checkpoint_dir}")


if __name__ == '__main__':
    main()


