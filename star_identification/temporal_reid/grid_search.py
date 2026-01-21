#!/usr/bin/env python
"""
Grid search for temporal re-identification hyperparameters.

Focuses on loss function configurations to find optimal settings.

Usage:
    python -m temporal_reid.grid_search --dataset-root ./star_dataset --epochs 15
    
Multi-GPU parallel execution:
    python -m temporal_reid.grid_search --dataset-root ./star_dataset --epochs 15 --parallel-gpus 0,1
"""
import argparse
import subprocess
import sys
import json
import time
import os
import threading
import queue
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional


# Define grid search configurations
GRID_CONFIGS = [
    # === Single Loss Baselines ===
    {
        "name": "circle_only",
        "desc": "Circle loss only (no triplet)",
        "loss.circle_weight": 1.0,
        "loss.triplet_weight": 0.0,
        "loss.circle_margin": 0.25,
    },
    {
        "name": "circle_only_m40",
        "desc": "Circle loss only with higher margin (0.4)",
        "loss.circle_weight": 1.0,
        "loss.triplet_weight": 0.0,
        "loss.circle_margin": 0.40,
    },
    {
        "name": "triplet_only",
        "desc": "Triplet loss only (no circle)",
        "loss.circle_weight": 0.0,
        "loss.triplet_weight": 1.0,
        "loss.triplet_margin": 0.3,
    },
    {
        "name": "triplet_only_m50",
        "desc": "Triplet loss only with higher margin (0.5)",
        "loss.circle_weight": 0.0,
        "loss.triplet_weight": 1.0,
        "loss.triplet_margin": 0.5,
    },
    
    # === Combined Loss Configurations ===
    {
        "name": "balanced",
        "desc": "Equal weight circle + triplet",
        "loss.circle_weight": 0.5,
        "loss.triplet_weight": 0.5,
        "loss.circle_margin": 0.25,
        "loss.triplet_margin": 0.3,
    },
    {
        "name": "triplet_heavy",
        "desc": "Triplet-dominant (70/30)",
        "loss.circle_weight": 0.3,
        "loss.triplet_weight": 0.7,
        "loss.circle_margin": 0.25,
        "loss.triplet_margin": 0.3,
    },
    {
        "name": "triplet_heavy_m50",
        "desc": "Triplet-dominant with higher triplet margin",
        "loss.circle_weight": 0.3,
        "loss.triplet_weight": 0.7,
        "loss.circle_margin": 0.25,
        "loss.triplet_margin": 0.5,
    },
    {
        "name": "circle_heavy_m40",
        "desc": "Circle-dominant (70/30) with higher circle margin",
        "loss.circle_weight": 0.7,
        "loss.triplet_weight": 0.3,
        "loss.circle_margin": 0.40,
        "loss.triplet_margin": 0.3,
    },
    
    # === Learning Rate Variations (with triplet-only since it showed signal) ===
    {
        "name": "triplet_lr5e5",
        "desc": "Triplet only with lower LR (5e-5)",
        "loss.circle_weight": 0.0,
        "loss.triplet_weight": 1.0,
        "loss.triplet_margin": 0.3,
        "learning_rate": 5e-5,
    },
    {
        "name": "triplet_lr2e4",
        "desc": "Triplet only with higher LR (2e-4)",
        "loss.circle_weight": 0.0,
        "loss.triplet_weight": 1.0,
        "loss.triplet_margin": 0.3,
        "learning_rate": 2e-4,
    },
]


def create_config_file(base_config_path: Path, overrides: Dict[str, Any], output_path: Path) -> Path:
    """Create a config file with overrides applied."""
    from .config import Config, LossConfig
    
    # Load base config or create default
    if base_config_path and base_config_path.exists():
        config = Config.load(base_config_path)
    else:
        config = Config()
    
    # Apply overrides
    for key, value in overrides.items():
        if key in ("name", "desc"):
            continue
        
        if key.startswith("loss."):
            loss_key = key.replace("loss.", "")
            setattr(config.loss, loss_key, value)
        else:
            setattr(config, key, value)
    
    # Save config
    config.save(output_path)
    return output_path


def run_training(
    config_path: Path,
    dataset_root: str,
    epochs: int,
    checkpoint_dir: Path,
    batch_size: int = 16,
    gpu_id: Optional[int] = None,
    gpus: Optional[str] = None,
    log_file: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run a single training experiment and return results."""
    
    cmd = [
        sys.executable, "-m", "temporal_reid.train",
        "--config", str(config_path),
        "--dataset-root", dataset_root,
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--checkpoint-dir", str(checkpoint_dir),
    ]
    
    # Add multi-GPU DataParallel flag if specified
    if gpus:
        cmd.extend(["--gpus", gpus])
    
    # Set environment for specific GPU
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"  # Ensure UTF-8 output on Windows
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        gpu_str = f"[GPU {gpu_id}] "
    else:
        gpu_str = ""
    
    print(f"\n{'='*60}")
    print(f"{gpu_str}Running: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    try:
        if log_file:
            # Write output to log file for parallel execution
            with open(log_file, 'w', encoding='utf-8') as f:
                result = subprocess.run(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding='utf-8',
                    cwd=Path(__file__).parent.parent,
                    env=env,
                )
        else:
            result = subprocess.run(
                cmd,
                capture_output=False,
                text=True,
                cwd=Path(__file__).parent.parent,
                env=env,
            )
        success = result.returncode == 0
    except Exception as e:
        print(f"{gpu_str}Error running training: {e}")
        success = False
    
    elapsed = time.time() - start_time
    
    # Try to load results from checkpoint
    metrics = {}
    best_path = checkpoint_dir / "best.pth"
    if best_path.exists():
        import torch
        checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
        metrics = checkpoint.get("metrics", {})
    
    return {
        "success": success,
        "elapsed_seconds": elapsed,
        "metrics": metrics,
    }


def gpu_worker(
    gpu_id: int,
    task_queue: queue.Queue,
    result_list: list,
    result_lock: threading.Lock,
    dataset_root: str,
    epochs: int,
    batch_size: int,
    output_dir: Path,
):
    """Worker thread that processes experiments on a specific GPU."""
    while True:
        try:
            task = task_queue.get_nowait()
        except queue.Empty:
            break
        
        idx, cfg = task
        name = cfg["name"]
        
        print(f"\n[GPU {gpu_id}] Starting experiment {idx}: {name}")
        
        # Create config file
        config_dir = output_dir / name
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / "config.json"
        log_file = config_dir / "training.log"
        
        create_config_file(None, cfg, config_path)
        
        # Run training
        checkpoint_dir = config_dir / "checkpoints"
        result = run_training(
            config_path=config_path,
            dataset_root=dataset_root,
            epochs=epochs,
            checkpoint_dir=checkpoint_dir,
            batch_size=batch_size,
            gpu_id=gpu_id,
            log_file=log_file,
        )
        
        # Store result
        result["name"] = name
        result["desc"] = cfg["desc"]
        result["config"] = {k: v for k, v in cfg.items() if k not in ("name", "desc")}
        result["gpu_id"] = gpu_id
        
        with result_lock:
            result_list.append(result)
        
        print(f"[GPU {gpu_id}] Completed {name}: mAP={result['metrics'].get('mAP', 'N/A')}")
        task_queue.task_done()


def main():
    parser = argparse.ArgumentParser(description='Grid search for temporal re-ID')
    parser.add_argument('--dataset-root', type=str, default='./star_dataset',
                        help='Path to star dataset')
    parser.add_argument('--epochs', type=int, default=15,
                        help='Epochs per experiment')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--output-dir', type=str, default='./grid_search_results',
                        help='Output directory for results')
    parser.add_argument('--base-config', type=str, default=None,
                        help='Base config to modify')
    parser.add_argument('--configs', type=str, nargs='+', default=None,
                        help='Specific config names to run (default: all)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print configs without running')
    parser.add_argument('--parallel-gpus', type=str, default=None,
                        help='Comma-separated GPU IDs for parallel execution (e.g., "0,1")')
    parser.add_argument('--gpus', type=str, default=None,
                        help='Use DataParallel on these GPUs for each training run (e.g., "0,1")')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Filter configs if specified
    configs_to_run = GRID_CONFIGS
    if args.configs:
        configs_to_run = [c for c in GRID_CONFIGS if c["name"] in args.configs]
        if not configs_to_run:
            print(f"No matching configs found. Available: {[c['name'] for c in GRID_CONFIGS]}")
            return
    
    print("=" * 60)
    print("GRID SEARCH - TEMPORAL RE-IDENTIFICATION")
    print("=" * 60)
    print(f"\nDataset: {args.dataset_root}")
    print(f"Epochs per run: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    if args.gpus:
        print(f"DataParallel GPUs: {args.gpus}")
    print(f"Output: {output_dir}")
    print(f"\nConfigurations to run ({len(configs_to_run)}):")
    
    for i, cfg in enumerate(configs_to_run, 1):
        print(f"  {i}. {cfg['name']}: {cfg['desc']}")
        if args.dry_run:
            for k, v in cfg.items():
                if k not in ("name", "desc"):
                    print(f"      {k} = {v}")
    
    if args.dry_run:
        print("\n[DRY RUN - not executing]")
        return
    
    # Parse GPU IDs for parallel execution
    gpu_ids = None
    if args.parallel_gpus:
        gpu_ids = [int(g.strip()) for g in args.parallel_gpus.split(",")]
        print(f"\nParallel execution on GPUs: {gpu_ids}")
        print(f"Estimated time: ~{len(configs_to_run) * 25 / len(gpu_ids):.0f} minutes")
    else:
        print(f"\nEstimated time: ~{len(configs_to_run) * 25} minutes")
    
    print("\nStarting in 5 seconds... (Ctrl+C to cancel)")
    time.sleep(5)
    
    results = []
    
    if gpu_ids and len(gpu_ids) > 1:
        # === PARALLEL EXECUTION ===
        print(f"\n{'='*60}")
        print(f"PARALLEL EXECUTION ON {len(gpu_ids)} GPUs")
        print(f"{'='*60}")
        
        # Create task queue
        task_queue = queue.Queue()
        for i, cfg in enumerate(configs_to_run, 1):
            task_queue.put((i, cfg))
        
        result_lock = threading.Lock()
        
        # Start worker threads
        threads = []
        for gpu_id in gpu_ids:
            t = threading.Thread(
                target=gpu_worker,
                args=(
                    gpu_id,
                    task_queue,
                    results,
                    result_lock,
                    args.dataset_root,
                    args.epochs,
                    args.batch_size,
                    output_dir,
                ),
            )
            t.start()
            threads.append(t)
        
        # Wait for all to complete
        for t in threads:
            t.join()
        
        # Save results
        results_path = output_dir / f"results_{timestamp}.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
    
    else:
        # === SEQUENTIAL EXECUTION ===
        gpu_id = gpu_ids[0] if gpu_ids else None
        
        for i, cfg in enumerate(configs_to_run, 1):
            name = cfg["name"]
            print(f"\n{'#'*60}")
            print(f"# Experiment {i}/{len(configs_to_run)}: {name}")
            print(f"# {cfg['desc']}")
            print(f"{'#'*60}")
            
            # Create config file
            config_dir = output_dir / name
            config_dir.mkdir(parents=True, exist_ok=True)
            config_path = config_dir / "config.json"
            
            base_config = Path(args.base_config) if args.base_config else None
            create_config_file(base_config, cfg, config_path)
            
            # Run training
            checkpoint_dir = config_dir / "checkpoints"
            result = run_training(
                config_path=config_path,
                dataset_root=args.dataset_root,
                epochs=args.epochs,
                checkpoint_dir=checkpoint_dir,
                batch_size=args.batch_size,
                gpu_id=gpu_id,
                gpus=args.gpus,
            )
            
            # Store result
            result["name"] = name
            result["desc"] = cfg["desc"]
            result["config"] = {k: v for k, v in cfg.items() if k not in ("name", "desc")}
            results.append(result)
            
            # Save intermediate results
            results_path = output_dir / f"results_{timestamp}.json"
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"\nCompleted {name}:")
            print(f"  Time: {result['elapsed_seconds']/60:.1f} min")
            if result['metrics']:
                mAP = result['metrics'].get('mAP')
                rank1 = result['metrics'].get('rank_1')
                if mAP is not None:
                    print(f"  mAP: {mAP:.4f}")
                if rank1 is not None:
                    print(f"  Rank-1: {rank1:.4f}")
    
    # Final summary
    print("\n" + "=" * 60)
    print("GRID SEARCH COMPLETE")
    print("=" * 60)
    
    # Sort by mAP
    valid_results = [r for r in results if r.get('metrics', {}).get('mAP')]
    valid_results.sort(key=lambda x: x['metrics']['mAP'], reverse=True)
    
    print("\nResults ranked by mAP:")
    print("-" * 60)
    print(f"{'Rank':<5} {'Name':<25} {'mAP':>8} {'Rank-1':>8} {'Time':>8}")
    print("-" * 60)
    
    for i, r in enumerate(valid_results, 1):
        m = r['metrics']
        print(f"{i:<5} {r['name']:<25} {m.get('mAP', 0):.4f}  {m.get('rank_1', 0):.4f}  {r['elapsed_seconds']/60:.1f}m")
    
    print("-" * 60)
    print(f"\nFull results saved to: {results_path}")
    
    # Save summary
    summary_path = output_dir / f"summary_{timestamp}.txt"
    with open(summary_path, 'w') as f:
        f.write("Grid Search Results\n")
        f.write("=" * 60 + "\n\n")
        for i, r in enumerate(valid_results, 1):
            m = r['metrics']
            f.write(f"{i}. {r['name']}\n")
            f.write(f"   {r['desc']}\n")
            f.write(f"   mAP: {m.get('mAP', 0):.4f}, Rank-1: {m.get('rank_1', 0):.4f}\n")
            f.write(f"   Config: {r['config']}\n\n")
    
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()

