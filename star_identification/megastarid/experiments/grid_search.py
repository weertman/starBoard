#!/usr/bin/env python
"""
Grid Search Experiment for MegaStarID.

Tests combinations of:
- Training strategy: pretrain→finetune, cotrain, star-only (baseline)
- Loss function: circle-only, triplet-only

Runs sequentially using both GPUs via DataParallel.

Usage:
    python -m megastarid.experiments.grid_search
    python -m megastarid.experiments.grid_search --epochs 25 --gpus 0,1
    python -m megastarid.experiments.grid_search --dry-run
"""
import argparse
import json
import time
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict, field

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

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
from megastarid.models import create_model, load_pretrained_model, count_parameters
from megastarid.trainer import MegaStarTrainer

from wildlife_reid.datasets import Wildlife10kDataset
from wildlife_reid.config import Wildlife10kConfig, FilterConfig, SplitConfig
from wildlife_reid.utils.transforms import get_test_transforms


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    name: str
    strategy: str  # 'pretrain_finetune', 'cotrain', 'star_only'
    loss_type: str  # 'circle', 'triplet'
    
    # Epochs
    pretrain_epochs: int = 25
    finetune_epochs: int = 25
    cotrain_epochs: int = 25
    
    # Batch settings
    batch_size: int = 32
    num_instances: int = 4
    
    # For cotrain
    star_batch_ratio: float = 0.3


@dataclass
class ExperimentResult:
    """Results from a single experiment."""
    name: str
    strategy: str
    loss_type: str
    
    # Final metrics on star_dataset
    star_mAP: float
    star_rank1: float
    star_rank5: float
    
    # Training info
    total_time_seconds: float
    best_epoch: int
    
    # All star metrics
    all_metrics: Dict[str, float] = None
    
    # Per-subdataset Wildlife10k metrics (only for pretrain_finetune and cotrain)
    wildlife_per_dataset: Dict[str, Dict[str, float]] = None


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


def compute_per_dataset_wildlife_metrics(
    model: nn.Module,
    device: torch.device,
    wildlife_root: str = './wildlifeReID',
    image_size: int = 384,
    batch_size: int = 64,
    num_workers: int = 4,
) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics for each sub-dataset in Wildlife10k.
    
    Returns:
        Dict mapping dataset name to metrics dict (mAP, Rank-1, Rank-5, Rank-10)
    """
    print("\n" + "-" * 60)
    print("FINAL WILDLIFE10K PER-DATASET EVALUATION")
    print("-" * 60)
    
    model.eval()
    test_transform = get_test_transforms(image_size)
    
    # Load full Wildlife10k test set
    config = Wildlife10kConfig(
        data_root=wildlife_root,
        filter=FilterConfig(exclude_datasets=['SeaStarReID2023']),
        split=SplitConfig(strategy='original'),
        image_size=image_size,
    )
    
    test_dataset = Wildlife10kDataset(
        data_root=wildlife_root,
        transform=test_transform,
        mode='test',
        config=config,
    )
    
    # Get unique datasets
    unique_datasets = test_dataset.df['dataset'].unique()
    print(f"Evaluating on {len(unique_datasets)} sub-datasets...")
    
    results = {}
    
    for ds_name in sorted(unique_datasets):
        # Get subset for this dataset
        ds_df = test_dataset.df[test_dataset.df['dataset'] == ds_name]
        
        if len(ds_df) < 4:  # Need enough samples
            print(f"  {ds_name}: skipped (only {len(ds_df)} samples)")
            continue
        
        # Get indices for this dataset
        ds_indices = ds_df.index.tolist()
        
        # Extract embeddings for this dataset
        embeddings = []
        labels = []
        
        with torch.no_grad():
            for idx in ds_indices:
                sample = test_dataset[idx]
                img = sample['image'].unsqueeze(0).to(device)
                emb = model(img, return_normalized=True)
                embeddings.append(emb.cpu())
                labels.append(sample['label'])
        
        embeddings = torch.cat(embeddings, dim=0).numpy()
        labels = np.array(labels)
        
        # Compute similarities
        similarities = embeddings @ embeddings.T
        
        # Compute metrics
        all_aps = []
        all_ranks = []
        
        for i in range(len(labels)):
            query_label = labels[i]
            sims = similarities[i].copy()
            sims[i] = -np.inf  # Exclude self
            
            sorted_indices = np.argsort(-sims)
            sorted_labels = labels[sorted_indices]
            matches = sorted_labels == query_label
            
            if matches.sum() == 0:
                continue
            
            # AP
            cumsum = np.cumsum(matches)
            precision = cumsum / (np.arange(len(matches)) + 1)
            ap = (precision * matches).sum() / matches.sum()
            all_aps.append(ap)
            
            # Rank
            first_match = np.where(matches)[0][0]
            all_ranks.append(first_match + 1)
        
        if all_aps:
            metrics = {
                'mAP': float(np.mean(all_aps)),
                'Rank-1': float(np.mean([1.0 if r <= 1 else 0.0 for r in all_ranks])),
                'Rank-5': float(np.mean([1.0 if r <= 5 else 0.0 for r in all_ranks])),
                'Rank-10': float(np.mean([1.0 if r <= 10 else 0.0 for r in all_ranks])),
                'num_samples': len(ds_df),
                'num_identities': ds_df['identity'].nunique(),
            }
            results[ds_name] = metrics
            print(f"  {ds_name}: mAP={metrics['mAP']:.4f}, R1={metrics['Rank-1']:.4f} ({len(ds_df)} samples)")
    
    # Compute overall average
    if results:
        avg_mAP = np.mean([r['mAP'] for r in results.values()])
        avg_r1 = np.mean([r['Rank-1'] for r in results.values()])
        print(f"\n  Average across {len(results)} datasets: mAP={avg_mAP:.4f}, R1={avg_r1:.4f}")
        results['_average'] = {
            'mAP': float(avg_mAP),
            'Rank-1': float(avg_r1),
            'num_datasets': len(results),
        }
    
    return results


def run_pretrain_finetune(
    exp: ExperimentConfig,
    gpu_ids: List[int],
    base_dir: Path,
    seed: int = 42,
    num_workers: int = 4,
) -> ExperimentResult:
    """Run pretrain → finetune experiment."""
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {exp.name}")
    print(f"Strategy: Pretrain → Finetune | Loss: {exp.loss_type}")
    print(f"{'='*60}")
    
    start_time = time.time()
    device = torch.device(f'cuda:{gpu_ids[0]}' if torch.cuda.is_available() else 'cpu')
    
    exp_dir = base_dir / exp.name
    pretrain_dir = exp_dir / 'pretrain'
    finetune_dir = exp_dir / 'finetune'
    
    # ===== PHASE 1: Pre-training on Wildlife10k =====
    print("\n--- Phase 1: Pre-training on Wildlife10k ---")
    
    pretrain_config = PretrainConfig(
        checkpoint_dir=str(pretrain_dir),
        model=ModelConfig(),
        loss=get_loss_config(exp.loss_type),
        num_epochs=exp.pretrain_epochs,
        batch_size=exp.batch_size,
        num_instances=exp.num_instances,
        num_workers=num_workers,
        seed=seed,
        exclude_datasets=['SeaStarReID2023'],
        val_every_n_epochs=5,
    )
    
    # Load data
    train_loader, test_loader = create_pretrain_dataloaders(pretrain_config)
    
    # Create model with DataParallel if multiple GPUs
    model = create_model(pretrain_config)
    if len(gpu_ids) > 1:
        print(f"Using DataParallel on GPUs: {gpu_ids}")
        model = nn.DataParallel(model, device_ids=gpu_ids)
    model = model.to(device)
    
    # Train
    trainer = MegaStarTrainer(model, pretrain_config, device, mode='pretrain')
    trainer.train_pretrain(train_loader, test_loader, exp.pretrain_epochs, val_every=5)
    
    pretrain_checkpoint = pretrain_dir / 'best.pth'
    
    # ===== PHASE 2: Fine-tuning on star_dataset =====
    print("\n--- Phase 2: Fine-tuning on star_dataset ---")
    
    finetune_config = FinetuneConfig(
        checkpoint_dir=str(finetune_dir),
        pretrain_checkpoint=str(pretrain_checkpoint),
        model=ModelConfig(),
        loss=get_loss_config(exp.loss_type),
        num_epochs=exp.finetune_epochs,
        batch_size=exp.batch_size,
        num_instances=exp.num_instances,
        learning_rate=5e-5,  # Lower for fine-tuning
        num_workers=num_workers,
        seed=seed,
        val_every_n_epochs=5,
    )
    
    # Load data
    train_loader, gallery_loader, query_loader = create_finetune_dataloaders(finetune_config)
    
    # Load pre-trained model
    model = load_pretrained_model(finetune_config, str(pretrain_checkpoint), device)
    if len(gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=gpu_ids)
    model = model.to(device)
    
    # Train
    trainer = MegaStarTrainer(model, finetune_config, device, mode='finetune')
    best_metrics = trainer.train_finetune(
        train_loader, gallery_loader, query_loader,
        exp.finetune_epochs, val_every=5
    )
    
    # ===== FINAL: Per-dataset Wildlife10k evaluation =====
    # Get the underlying model (unwrap DataParallel if needed)
    eval_model = model.module if isinstance(model, nn.DataParallel) else model
    wildlife_metrics = compute_per_dataset_wildlife_metrics(
        eval_model, device, 
        batch_size=exp.batch_size,
        num_workers=num_workers,
    )
    
    # Save wildlife metrics to file
    with open(exp_dir / 'wildlife_per_dataset.json', 'w') as f:
        json.dump(wildlife_metrics, f, indent=2)
    
    total_time = time.time() - start_time
    
    return ExperimentResult(
        name=exp.name,
        strategy='pretrain_finetune',
        loss_type=exp.loss_type,
        star_mAP=best_metrics.get('mAP', 0),
        star_rank1=best_metrics.get('Rank-1', 0),
        star_rank5=best_metrics.get('Rank-5', 0),
        total_time_seconds=total_time,
        best_epoch=trainer.best_epoch,
        all_metrics=best_metrics,
        wildlife_per_dataset=wildlife_metrics,
    )


def run_cotrain(
    exp: ExperimentConfig,
    gpu_ids: List[int],
    base_dir: Path,
    seed: int = 42,
    num_workers: int = 4,
) -> ExperimentResult:
    """Run co-training experiment."""
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {exp.name}")
    print(f"Strategy: Co-training | Loss: {exp.loss_type}")
    print(f"{'='*60}")
    
    start_time = time.time()
    device = torch.device(f'cuda:{gpu_ids[0]}' if torch.cuda.is_available() else 'cpu')
    
    exp_dir = base_dir / exp.name
    
    config = CotrainConfig(
        checkpoint_dir=str(exp_dir),
        model=ModelConfig(),
        loss=get_loss_config(exp.loss_type),
        num_epochs=exp.cotrain_epochs,
        batch_size=exp.batch_size,
        num_instances=exp.num_instances,
        star_batch_ratio=exp.star_batch_ratio,
        num_workers=num_workers,
        seed=seed,
        exclude_datasets=['SeaStarReID2023'],
        val_every_n_epochs=5,
    )
    
    # Load data
    train_loader, wildlife_test, star_gallery, star_query = create_cotrain_dataloaders(config)
    
    # Create model
    model = create_model(config)
    if len(gpu_ids) > 1:
        print(f"Using DataParallel on GPUs: {gpu_ids}")
        model = nn.DataParallel(model, device_ids=gpu_ids)
    model = model.to(device)
    
    # Train
    trainer = MegaStarTrainer(model, config, device, mode='cotrain')
    best_metrics = trainer.train_cotrain(
        train_loader, wildlife_test, star_gallery, star_query,
        exp.cotrain_epochs, val_every=5
    )
    
    # ===== FINAL: Per-dataset Wildlife10k evaluation =====
    eval_model = model.module if isinstance(model, nn.DataParallel) else model
    wildlife_metrics = compute_per_dataset_wildlife_metrics(
        eval_model, device,
        batch_size=exp.batch_size,
        num_workers=num_workers,
    )
    
    # Save wildlife metrics to file
    with open(exp_dir / 'wildlife_per_dataset.json', 'w') as f:
        json.dump(wildlife_metrics, f, indent=2)
    
    total_time = time.time() - start_time
    
    return ExperimentResult(
        name=exp.name,
        strategy='cotrain',
        loss_type=exp.loss_type,
        star_mAP=best_metrics.get('mAP', 0),
        star_rank1=best_metrics.get('Rank-1', 0),
        star_rank5=best_metrics.get('Rank-5', 0),
        total_time_seconds=total_time,
        best_epoch=trainer.best_epoch,
        all_metrics=best_metrics,
        wildlife_per_dataset=wildlife_metrics,
    )


def run_star_only(
    exp: ExperimentConfig,
    gpu_ids: List[int],
    base_dir: Path,
    seed: int = 42,
    num_workers: int = 4,
) -> ExperimentResult:
    """Run star-only baseline experiment."""
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {exp.name}")
    print(f"Strategy: Star-only (baseline) | Loss: {exp.loss_type}")
    print(f"{'='*60}")
    
    start_time = time.time()
    device = torch.device(f'cuda:{gpu_ids[0]}' if torch.cuda.is_available() else 'cpu')
    
    exp_dir = base_dir / exp.name
    
    config = FinetuneConfig(
        checkpoint_dir=str(exp_dir),
        pretrain_checkpoint=None,  # No pre-training
        model=ModelConfig(),
        loss=get_loss_config(exp.loss_type),
        num_epochs=exp.finetune_epochs,
        batch_size=exp.batch_size,
        num_instances=exp.num_instances,
        learning_rate=1e-4,  # Standard LR for training from scratch
        num_workers=num_workers,
        seed=seed,
        val_every_n_epochs=5,
    )
    
    # Load data
    train_loader, gallery_loader, query_loader = create_finetune_dataloaders(config)
    
    # Create model (from scratch)
    model = create_model(config)
    if len(gpu_ids) > 1:
        print(f"Using DataParallel on GPUs: {gpu_ids}")
        model = nn.DataParallel(model, device_ids=gpu_ids)
    model = model.to(device)
    
    # Train
    trainer = MegaStarTrainer(model, config, device, mode='finetune')
    best_metrics = trainer.train_finetune(
        train_loader, gallery_loader, query_loader,
        exp.finetune_epochs, val_every=5
    )
    
    total_time = time.time() - start_time
    
    # No wildlife metrics for star-only experiments
    return ExperimentResult(
        name=exp.name,
        strategy='star_only',
        loss_type=exp.loss_type,
        star_mAP=best_metrics.get('mAP', 0),
        star_rank1=best_metrics.get('Rank-1', 0),
        star_rank5=best_metrics.get('Rank-5', 0),
        total_time_seconds=total_time,
        best_epoch=trainer.best_epoch,
        all_metrics=best_metrics,
        wildlife_per_dataset=None,
    )


def create_experiment_grid(epochs: int = 25, batch_size: int = 32) -> List[ExperimentConfig]:
    """Create grid of experiments to run."""
    experiments = []
    
    # Baseline: Star-only
    for loss in ['circle', 'triplet']:
        experiments.append(ExperimentConfig(
            name=f'star_only_{loss}',
            strategy='star_only',
            loss_type=loss,
            finetune_epochs=epochs,
            batch_size=batch_size,
        ))
    
    # Pretrain → Finetune
    for loss in ['circle', 'triplet']:
        experiments.append(ExperimentConfig(
            name=f'pretrain_finetune_{loss}',
            strategy='pretrain_finetune',
            loss_type=loss,
            pretrain_epochs=epochs,
            finetune_epochs=epochs,
            batch_size=batch_size,
        ))
    
    # Co-training
    for loss in ['circle', 'triplet']:
        experiments.append(ExperimentConfig(
            name=f'cotrain_{loss}',
            strategy='cotrain',
            loss_type=loss,
            cotrain_epochs=epochs,
            batch_size=batch_size,
        ))
    
    return experiments


def run_experiment(
    exp: ExperimentConfig,
    gpu_ids: List[int],
    base_dir: Path,
    seed: int,
    num_workers: int,
) -> ExperimentResult:
    """Run a single experiment based on its strategy."""
    if exp.strategy == 'pretrain_finetune':
        return run_pretrain_finetune(exp, gpu_ids, base_dir, seed, num_workers)
    elif exp.strategy == 'cotrain':
        return run_cotrain(exp, gpu_ids, base_dir, seed, num_workers)
    elif exp.strategy == 'star_only':
        return run_star_only(exp, gpu_ids, base_dir, seed, num_workers)
    else:
        raise ValueError(f"Unknown strategy: {exp.strategy}")


def save_results(results: List[ExperimentResult], output_path: Path):
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


def print_summary(results: List[ExperimentResult]):
    """Print summary table of results."""
    print("\n" + "=" * 80)
    print("EXPERIMENT RESULTS SUMMARY (Star Dataset)")
    print("=" * 80)
    print(f"{'Experiment':<30} {'mAP':>8} {'Rank-1':>8} {'Rank-5':>8} {'Time':>10}")
    print("-" * 80)
    
    # Sort by mAP
    sorted_results = sorted(results, key=lambda x: x.star_mAP, reverse=True)
    
    for r in sorted_results:
        time_str = f"{r.total_time_seconds/60:.1f}m"
        print(f"{r.name:<30} {r.star_mAP:>8.4f} {r.star_rank1:>8.4f} {r.star_rank5:>8.4f} {time_str:>10}")
    
    print("-" * 80)
    
    # Best result
    best = sorted_results[0]
    print(f"\n🏆 Best: {best.name} with mAP={best.star_mAP:.4f}")
    
    # Wildlife summary for experiments that have it
    wildlife_exps = [r for r in results if r.wildlife_per_dataset]
    if wildlife_exps:
        print("\n" + "=" * 80)
        print("WILDLIFE10K PER-DATASET SUMMARY")
        print("=" * 80)
        for r in wildlife_exps:
            if '_average' in r.wildlife_per_dataset:
                avg = r.wildlife_per_dataset['_average']
                print(f"{r.name}: avg mAP={avg['mAP']:.4f} across {avg['num_datasets']} datasets")


def main():
    parser = argparse.ArgumentParser(description='Grid search experiments')
    
    parser.add_argument('--epochs', type=int, default=25,
                        help='Epochs per training phase')
    parser.add_argument('--gpus', type=str, default='0,1',
                        help='Comma-separated GPU IDs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='DataLoader workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--output-dir', type=str,
                        default='./checkpoints/megastarid/grid_search',
                        help='Output directory')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print experiment plan without running')
    parser.add_argument('--experiments', type=str, nargs='+', default=None,
                        help='Run only specific experiments (by name)')
    
    args = parser.parse_args()
    
    # Parse GPU IDs
    gpu_ids = [int(x) for x in args.gpus.split(',')]
    print(f"Using GPUs: {gpu_ids}")
    
    # Set seed
    set_seed(args.seed)
    
    # Create experiment grid
    experiments = create_experiment_grid(args.epochs, args.batch_size)
    
    # Filter if specific experiments requested
    if args.experiments:
        experiments = [e for e in experiments if e.name in args.experiments]
    
    # Print plan
    print("\n" + "=" * 60)
    print("GRID SEARCH EXPERIMENT PLAN")
    print("=" * 60)
    print(f"Epochs per phase: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"GPUs: {gpu_ids}")
    print(f"\nExperiments to run ({len(experiments)}):")
    for i, exp in enumerate(experiments, 1):
        print(f"  {i}. {exp.name} ({exp.strategy}, {exp.loss_type})")
    
    if args.dry_run:
        print("\n[DRY RUN - not executing]")
        return
    
    # Run experiments
    base_dir = Path(args.output_dir)
    results = []
    
    for i, exp in enumerate(experiments, 1):
        print(f"\n\n{'#'*60}")
        print(f"# EXPERIMENT {i}/{len(experiments)}: {exp.name}")
        print(f"{'#'*60}")
        
        try:
            result = run_experiment(
                exp, gpu_ids, base_dir, args.seed, args.num_workers
            )
            results.append(result)
            
            # Save intermediate results
            save_results(results, base_dir / 'results.json')
            
        except Exception as e:
            print(f"\n❌ Experiment {exp.name} failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Final summary
    if results:
        print_summary(results)
        save_results(results, base_dir / 'results_final.json')


if __name__ == '__main__':
    main()
