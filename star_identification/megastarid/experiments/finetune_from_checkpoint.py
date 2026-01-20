"""
Fine-tune from an existing checkpoint on star_dataset.

Usage:
    python -m megastarid.experiments.finetune_from_checkpoint \
        --checkpoint path/to/best.pth \
        --output-dir path/to/output \
        --epochs 100 \
        --name my_experiment
        
Example - fine-tune both pretrained and cotrained models:
    python -m megastarid.experiments.finetune_from_checkpoint \
        --checkpoint checkpoints/pretrain/best.pth \
        --checkpoint checkpoints/cotrain/best.pth \
        --epochs 100 \
        --gpu 1
"""

import argparse
import json
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn

from megastarid.config import FinetuneConfig, ModelConfig
from megastarid.datasets.combined import create_finetune_dataloaders
from megastarid.models import load_pretrained_model
from megastarid.trainer import MegaStarTrainer
from megastarid.experiments.arch_grid_search import (
    clear_gpu_memory, 
    HYPERPARAM_MAP,
    get_loss_config,
)
from megastarid.evaluation import evaluate_star_dataset


def run_finetune(
    checkpoint_path: Path,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    gpu_id: int,
    include_negative_only: bool,
    num_workers: int = 8,
    seed: int = 42,
):
    """Run fine-tuning from a checkpoint."""
    
    print(f"\n{'='*70}")
    print(f"FINE-TUNING FROM CHECKPOINT")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output: {output_dir}")
    print(f"Epochs: {epochs}")
    print(f"{'='*70}")
    
    clear_gpu_memory()
    start_time = time.time()
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load checkpoint to get model config
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Try to get architecture from checkpoint
    if 'config' in checkpoint:
        saved_config = checkpoint['config']
        # Try 'backbone' first (new format), then 'architecture' (old format)
        model_cfg = saved_config.get('model', {})
        architecture = model_cfg.get('backbone', model_cfg.get('architecture', 'convnext-tiny'))
    else:
        architecture = 'convnext-tiny'  # Default
    
    print(f"Architecture: {architecture}")
    
    # Get hyperparameters for this architecture
    hyperparams = HYPERPARAM_MAP.get(architecture, HYPERPARAM_MAP['convnext-tiny'])
    
    # Use lower LR for fine-tuning (half of pretrain LR)
    finetune_lr = hyperparams['learning_rate'] * 0.5
    
    # Model config
    model_config = ModelConfig(
        backbone=architecture,
        use_multiscale=True,
        use_bnneck=True,
        embedding_dim=512,
        pretrained=True,
    )
    
    # Fine-tune config
    finetune_config = FinetuneConfig(
        checkpoint_dir=str(output_dir.parent),  # FinetuneConfig adds /finetune
        pretrain_checkpoint=str(checkpoint_path),
        model=model_config,
        loss=get_loss_config('triplet'),
        num_epochs=epochs,
        batch_size=batch_size,
        num_instances=4,
        learning_rate=finetune_lr,
        weight_decay=hyperparams['weight_decay'],
        warmup_ratio=hyperparams['warmup_ratio'],
        grad_clip_norm=hyperparams['grad_clip_norm'],
        use_llrd=hyperparams['use_llrd'],
        llrd_decay=hyperparams['llrd_decay'],
        backbone_lr_mult=hyperparams['backbone_lr_mult'],
        num_workers=num_workers,
        seed=seed,
        include_negative_only=include_negative_only,
        val_every_n_epochs=max(1, epochs // 20),
    )
    
    # Override checkpoint_dir since FinetuneConfig modifies it
    finetune_config.checkpoint_dir = str(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"LR: {finetune_lr:.2e} | WD: {hyperparams['weight_decay']:.2e}")
    print(f"Include negative-only IDs: {include_negative_only}")
    
    # Create dataloaders
    train_loader, gallery_loader, query_loader = create_finetune_dataloaders(finetune_config)
    
    # Load model from checkpoint
    model = load_pretrained_model(finetune_config, str(checkpoint_path), device)
    model = model.to(device)
    
    # Train
    trainer = MegaStarTrainer(model, finetune_config, device, mode='finetune')
    best_metrics = trainer.train_finetune(
        train_loader, gallery_loader, query_loader,
        epochs, val_every=5, validate_first=True
    )
    
    total_time = time.time() - start_time
    
    print(f"\n✅ Fine-tuning complete! Best mAP: {best_metrics.get('mAP', 0):.4f}")
    print(f"Total time: {total_time/60:.1f} minutes")
    
    # Run detailed evaluation
    print("\nRunning detailed star_dataset evaluation (with TTA + reranking)...")
    best_checkpoint = output_dir / 'best.pth'
    
    if best_checkpoint.exists():
        # Reload best model for evaluation
        model = load_pretrained_model(finetune_config, str(best_checkpoint), device)
        model = model.to(device)
        model.eval()
        
        eval_results = evaluate_star_dataset(
            model=model,
            device=device,
            output_dir=output_dir,
            use_tta=True,
            use_reranking=True,
        )
        
        # Save results
        results = {
            'checkpoint': str(checkpoint_path),
            'output_dir': str(output_dir),
            'epochs': epochs,
            'architecture': architecture,
            'include_negative_only': include_negative_only,
            'best_val_mAP': best_metrics.get('mAP', 0),
            'best_val_rank1': best_metrics.get('Rank-1', 0),
            'final_mAP_tta_rerank': eval_results.get('mAP', 0),
            'final_rank1_tta_rerank': eval_results.get('CMC@1', 0),
            'total_time_minutes': total_time / 60,
        }
        
        with open(output_dir / 'finetune_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    return {'best_mAP': best_metrics.get('mAP', 0)}


def main():
    parser = argparse.ArgumentParser(description='Fine-tune from checkpoint(s)')
    parser.add_argument('--checkpoint', type=str, action='append', required=True,
                       help='Path to checkpoint file (can be specified multiple times)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Base output directory (default: auto-generate)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of fine-tuning epochs')
    parser.add_argument('--batch-size', type=int, default=160,
                       help='Batch size')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU ID to use')
    parser.add_argument('--include-negative-only', action='store_true',
                       help='Include negative-only identities in training')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show plan without executing')
    
    args = parser.parse_args()
    
    # Generate output directory if not provided
    if args.output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_dir = f'./checkpoints/megastarid/finetune_{timestamp}'
    
    base_output = Path(args.output_dir)
    
    print(f"\n{'='*70}")
    print("FINE-TUNING FROM CHECKPOINT(S)")
    print(f"{'='*70}")
    print(f"Output base: {base_output}")
    print(f"Epochs: {args.epochs}")
    print(f"GPU: {args.gpu}")
    print(f"Include negative-only: {args.include_negative_only}")
    print(f"\nCheckpoints to fine-tune ({len(args.checkpoint)}):")
    
    for i, ckpt in enumerate(args.checkpoint, 1):
        ckpt_path = Path(ckpt)
        exists = "✓" if ckpt_path.exists() else "✗ NOT FOUND"
        # Derive experiment name from path
        name = ckpt_path.parent.name
        if name in ['pretrain', 'cotrain', 'finetune']:
            name = ckpt_path.parent.parent.name
        print(f"  {i}. [{exists}] {ckpt}")
        print(f"      Name: {name}")
        print(f"      Output: {base_output / name}")
    
    if args.dry_run:
        print("\n[DRY RUN - not executing]")
        return
    
    # Verify all checkpoints exist
    missing = [c for c in args.checkpoint if not Path(c).exists()]
    if missing:
        print(f"\n❌ Missing checkpoints:")
        for m in missing:
            print(f"  - {m}")
        print("\nPlease ensure all checkpoints exist before running.")
        return
    
    print(f"\nStarting in 5 seconds... (Ctrl+C to cancel)")
    time.sleep(5)
    
    # Run fine-tuning for each checkpoint
    all_results = []
    for i, ckpt in enumerate(args.checkpoint, 1):
        ckpt_path = Path(ckpt)
        
        # Derive name from checkpoint path
        name = ckpt_path.parent.name
        if name in ['pretrain', 'cotrain', 'finetune']:
            name = ckpt_path.parent.parent.name
        
        output_dir = base_output / name / 'finetune'
        
        print(f"\n{'#'*70}")
        print(f"# EXPERIMENT {i}/{len(args.checkpoint)}: {name}")
        print(f"{'#'*70}")
        
        try:
            result = run_finetune(
                checkpoint_path=ckpt_path,
                output_dir=output_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                gpu_id=args.gpu,
                include_negative_only=args.include_negative_only,
            )
            result['name'] = name
            all_results.append(result)
        except Exception as e:
            print(f"\n❌ Experiment {name} failed: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({'name': name, 'error': str(e)})
    
    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    for r in all_results:
        if 'error' in r:
            print(f"  {r['name']}: FAILED - {r['error']}")
        else:
            print(f"  {r['name']}:")
            print(f"    Val mAP: {r.get('best_val_mAP', 0):.4f}")
            print(f"    Final mAP (TTA+rerank): {r.get('final_mAP_tta_rerank', 0):.4f}")
            print(f"    Final R1 (TTA+rerank): {r.get('final_rank1_tta_rerank', 0):.4f}")
    
    # Save combined results
    with open(base_output / 'all_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to: {base_output / 'all_results.json'}")


if __name__ == '__main__':
    main()

