#!/usr/bin/env python
"""
Plotting utilities for MegaStarID grid search experiments.

Provides:
- Per-run training curve plots
- Grid search comparison plots
"""
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# =============================================================================
# Color Schemes
# =============================================================================

# Architecture colors - distinctive, colorblind-friendly palette
ARCH_COLORS = {
    'swinv2-tiny': '#2E86AB',      # Steel blue
    'densenet121': '#A23B72',      # Raspberry
    'densenet169': '#F18F01',      # Orange
    'resnet50': '#C73E1D',         # Vermillion
    'convnext-tiny': '#3B1F2B',    # Dark purple
}

# Loss type patterns/secondary colors
LOSS_COLORS = {
    'circle': '#4ECDC4',    # Teal
    'triplet': '#FF6B6B',   # Coral
}

# Strategy markers
STRATEGY_MARKERS = {
    'star_only': 'o',
    'pretrain_finetune': 's',
    'cotrain': '^',
}

STRATEGY_COLORS = {
    'star_only': '#95D5B2',         # Mint green
    'pretrain_finetune': '#74C0FC', # Sky blue  
    'cotrain': '#DDA0DD',           # Plum
}


# =============================================================================
# Per-Run Plotting
# =============================================================================

def plot_training_history(
    history_path: Path,
    output_path: Optional[Path] = None,
    experiment_name: Optional[str] = None,
) -> Path:
    """
    Generate training curves for a single experiment.
    
    Creates a 2x2 figure with:
    - Training loss curve
    - Learning rate schedule
    - Validation metrics (mAP, Rank-1, Rank-5)
    - Loss vs Validation overlay
    
    Args:
        history_path: Path to training_history.json
        output_path: Where to save the plot (default: same dir as history)
        experiment_name: Name for the title (default: inferred from path)
        
    Returns:
        Path to saved plot
    """
    # Load history
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    columns = history['columns']
    epochs = columns['epoch']
    
    if output_path is None:
        output_path = history_path.parent / 'training_plots.png'
    
    if experiment_name is None:
        experiment_name = history_path.parent.name
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Training History: {experiment_name}', fontsize=14, fontweight='bold')
    
    # --- Plot 1: Training Loss ---
    ax1 = axes[0, 0]
    
    # Find all training loss columns
    loss_cols = [c for c in columns.keys() if c.startswith('train_')]
    
    for col in loss_cols:
        label = col.replace('train_', '').title()
        values = columns[col]
        if col == 'train_total':
            ax1.plot(epochs, values, 'b-', linewidth=2, label=label, zorder=10)
        else:
            ax1.plot(epochs, values, '--', alpha=0.7, linewidth=1.5, label=label)
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(epochs[0], epochs[-1])
    
    # --- Plot 2: Learning Rate Schedule ---
    ax2 = axes[0, 1]
    
    lr_values = columns.get('learning_rate', [])
    if lr_values:
        ax2.plot(epochs, lr_values, 'g-', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(epochs[0], epochs[-1])
    else:
        ax2.text(0.5, 0.5, 'No LR data', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Learning Rate Schedule')
    
    # --- Plot 3: Validation Metrics ---
    ax3 = axes[1, 0]
    
    # Find validation metric columns
    val_cols = {
        'val_mAP': ('mAP', '#2E86AB', '-'),
        'val_Rank-1': ('Rank-1', '#A23B72', '--'),
        'val_Rank-5': ('Rank-5', '#F18F01', ':'),
        'val_Rank-10': ('Rank-10', '#6B7280', ':'),
    }
    
    has_val_data = False
    best_epoch = None
    best_map = 0
    
    for col, (label, color, linestyle) in val_cols.items():
        if col in columns:
            values = columns[col]
            # Filter out None values
            valid_epochs = [e for e, v in zip(epochs, values) if v is not None]
            valid_values = [v for v in values if v is not None]
            
            if valid_values:
                has_val_data = True
                ax3.plot(valid_epochs, valid_values, linestyle, color=color, 
                        linewidth=2, marker='o', markersize=6, label=label)
                
                # Track best mAP
                if col == 'val_mAP':
                    best_idx = np.argmax(valid_values)
                    best_epoch = valid_epochs[best_idx]
                    best_map = valid_values[best_idx]
    
    if has_val_data:
        # Mark best epoch
        if best_epoch is not None:
            ax3.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.5, label=f'Best (epoch {best_epoch})')
            ax3.scatter([best_epoch], [best_map], color='green', s=150, marker='*', zorder=10)
        
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Metric Value')
        ax3.set_title('Validation Metrics')
        ax3.legend(loc='lower right')
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(epochs[0], epochs[-1])
        ax3.set_ylim(0, 1)
    else:
        ax3.text(0.5, 0.5, 'No validation data', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Validation Metrics')
    
    # --- Plot 4: Loss vs Validation Overlay ---
    ax4 = axes[1, 1]
    
    if 'train_total' in columns and has_val_data:
        # Plot loss on left axis
        color_loss = '#2E86AB'
        ax4.plot(epochs, columns['train_total'], color=color_loss, linewidth=2, label='Train Loss')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Training Loss', color=color_loss)
        ax4.tick_params(axis='y', labelcolor=color_loss)
        ax4.set_xlim(epochs[0], epochs[-1])
        
        # Plot mAP on right axis
        if 'val_mAP' in columns:
            ax4_twin = ax4.twinx()
            color_map = '#A23B72'
            
            valid_epochs = [e for e, v in zip(epochs, columns['val_mAP']) if v is not None]
            valid_values = [v for v in columns['val_mAP'] if v is not None]
            
            ax4_twin.plot(valid_epochs, valid_values, color=color_map, linewidth=2, 
                         marker='o', markersize=6, label='Val mAP')
            ax4_twin.set_ylabel('Validation mAP', color=color_map)
            ax4_twin.tick_params(axis='y', labelcolor=color_map)
            ax4_twin.set_ylim(0, 1)
        
        ax4.set_title('Loss vs Validation')
        ax4.grid(True, alpha=0.3)
        
        # Combined legend
        lines1, labels1 = ax4.get_legend_handles_labels()
        if 'ax4_twin' in dir():
            lines2, labels2 = ax4_twin.get_legend_handles_labels()
            ax4.legend(lines1 + lines2, labels1 + labels2, loc='center right')
        else:
            ax4.legend(loc='center right')
    else:
        ax4.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Loss vs Validation')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    return output_path


# =============================================================================
# Grid Search Comparison Plotting
# =============================================================================

@dataclass
class ExperimentResult:
    """Simplified result structure for plotting."""
    name: str
    architecture: str
    strategy: str
    loss_type: str
    total_params_millions: float
    batch_size: int
    star_mAP: float
    star_rank1: float
    star_rank5: float
    total_time_seconds: float
    best_epoch: int


def load_results_from_json(results_path: Path) -> List[ExperimentResult]:
    """Load experiment results from JSON file."""
    with open(results_path, 'r') as f:
        data = json.load(f)
    
    results = []
    for r in data.get('results', []):
        results.append(ExperimentResult(
            name=r['name'],
            architecture=r['architecture'],
            strategy=r['strategy'],
            loss_type=r['loss_type'],
            total_params_millions=r['total_params_millions'],
            batch_size=r['batch_size'],
            star_mAP=r['star_mAP'],
            star_rank1=r['star_rank1'],
            star_rank5=r['star_rank5'],
            total_time_seconds=r['total_time_seconds'],
            best_epoch=r['best_epoch'],
        ))
    
    return results


def plot_grid_search_summary(
    results: List[ExperimentResult],
    output_dir: Path,
    filename: str = 'grid_search_summary.png',
) -> Path:
    """
    Generate comparison plots for all grid search experiments.
    
    Creates a multi-panel figure with:
    1. Ranked bar chart of all experiments by mAP
    2. Grouped comparison by architecture × loss
    3. Grouped comparison by architecture × strategy
    4. Efficiency scatter (params vs performance)
    5. Top N experiments multi-metric comparison
    6. Training time comparison
    
    Args:
        results: List of experiment results
        output_dir: Directory to save the plot
        filename: Output filename
        
    Returns:
        Path to saved plot
    """
    if not results:
        print("No results to plot!")
        return None
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    
    # Sort results by mAP for ranking
    sorted_results = sorted(results, key=lambda x: x.star_mAP, reverse=True)
    
    # Determine layout based on number of unique strategies
    strategies = list(set(r.strategy for r in results))
    has_multiple_strategies = len(strategies) > 1
    
    # Create figure - 3x2 layout
    fig = plt.figure(figsize=(18, 16))
    
    # --- Plot 1: Ranked Bar Chart ---
    ax1 = fig.add_subplot(3, 2, 1)
    
    names = [r.name for r in sorted_results]
    maps = [r.star_mAP for r in sorted_results]
    colors = [ARCH_COLORS.get(r.architecture, '#888888') for r in sorted_results]
    
    y_pos = np.arange(len(names))
    bars = ax1.barh(y_pos, maps, color=colors, edgecolor='white', linewidth=0.5)
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(names, fontsize=8)
    ax1.invert_yaxis()
    ax1.set_xlabel('mAP')
    ax1.set_title('All Experiments Ranked by mAP', fontweight='bold')
    ax1.set_xlim(0, max(maps) * 1.15)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, maps)):
        ax1.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{val:.3f}', va='center', fontsize=8)
    
    # Highlight best
    bars[0].set_edgecolor('gold')
    bars[0].set_linewidth(3)
    
    # Legend for architectures
    arch_patches = [mpatches.Patch(color=c, label=a) for a, c in ARCH_COLORS.items() 
                   if a in [r.architecture for r in results]]
    ax1.legend(handles=arch_patches, loc='lower right', fontsize=8)
    
    # --- Plot 2: Architecture × Loss Comparison ---
    ax2 = fig.add_subplot(3, 2, 2)
    
    architectures = sorted(set(r.architecture for r in results))
    loss_types = sorted(set(r.loss_type for r in results))
    
    x = np.arange(len(architectures))
    width = 0.35
    
    for i, loss in enumerate(loss_types):
        maps_by_arch = []
        for arch in architectures:
            matching = [r.star_mAP for r in results if r.architecture == arch and r.loss_type == loss]
            maps_by_arch.append(np.mean(matching) if matching else 0)
        
        offset = (i - len(loss_types)/2 + 0.5) * width
        bars = ax2.bar(x + offset, maps_by_arch, width, label=loss.title(),
                      color=LOSS_COLORS.get(loss, '#888888'), edgecolor='white')
        
        # Add value labels
        for bar, val in zip(bars, maps_by_arch):
            if val > 0:
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    ax2.set_xlabel('Architecture')
    ax2.set_ylabel('mAP')
    ax2.set_title('mAP by Architecture × Loss Function', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([a.replace('densenet', 'dn').replace('swinv2-', 'sv2-').replace('convnext-', 'cn-') 
                         for a in architectures], fontsize=9)
    ax2.legend()
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # --- Plot 3: Architecture × Strategy Comparison ---
    ax3 = fig.add_subplot(3, 2, 3)
    
    if has_multiple_strategies:
        x = np.arange(len(architectures))
        width = 0.25
        
        for i, strategy in enumerate(sorted(strategies)):
            maps_by_arch = []
            for arch in architectures:
                matching = [r.star_mAP for r in results if r.architecture == arch and r.strategy == strategy]
                maps_by_arch.append(np.mean(matching) if matching else 0)
            
            offset = (i - len(strategies)/2 + 0.5) * width
            label = strategy.replace('_', ' ').title()
            bars = ax3.bar(x + offset, maps_by_arch, width, label=label,
                          color=STRATEGY_COLORS.get(strategy, '#888888'), edgecolor='white')
            
            for bar, val in zip(bars, maps_by_arch):
                if val > 0:
                    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f'{val:.3f}', ha='center', va='bottom', fontsize=7)
        
        ax3.set_xlabel('Architecture')
        ax3.set_ylabel('mAP')
        ax3.set_title('mAP by Architecture × Strategy', fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels([a.replace('densenet', 'dn').replace('swinv2-', 'sv2-').replace('convnext-', 'cn-') 
                             for a in architectures], fontsize=9)
        ax3.legend(fontsize=8)
        ax3.set_ylim(0, 1)
        ax3.grid(True, alpha=0.3, axis='y')
    else:
        # Single strategy - show Rank-1 comparison instead
        x = np.arange(len(architectures))
        width = 0.35
        
        for i, loss in enumerate(loss_types):
            rank1_by_arch = []
            for arch in architectures:
                matching = [r.star_rank1 for r in results if r.architecture == arch and r.loss_type == loss]
                rank1_by_arch.append(np.mean(matching) if matching else 0)
            
            offset = (i - len(loss_types)/2 + 0.5) * width
            bars = ax3.bar(x + offset, rank1_by_arch, width, label=loss.title(),
                          color=LOSS_COLORS.get(loss, '#888888'), edgecolor='white')
        
        ax3.set_xlabel('Architecture')
        ax3.set_ylabel('Rank-1')
        ax3.set_title('Rank-1 by Architecture × Loss Function', fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels([a.replace('densenet', 'dn').replace('swinv2-', 'sv2-').replace('convnext-', 'cn-') 
                             for a in architectures], fontsize=9)
        ax3.legend()
        ax3.set_ylim(0, 1)
        ax3.grid(True, alpha=0.3, axis='y')
    
    # --- Plot 4: Efficiency Scatter (Params vs Performance) ---
    ax4 = fig.add_subplot(3, 2, 4)
    
    for r in results:
        color = ARCH_COLORS.get(r.architecture, '#888888')
        marker = STRATEGY_MARKERS.get(r.strategy, 'o')
        ax4.scatter(r.total_params_millions, r.star_mAP, 
                   c=color, marker=marker, s=100, alpha=0.8, edgecolors='white', linewidth=1)
    
    # Add annotations for top 3
    for r in sorted_results[:3]:
        ax4.annotate(r.name.split('_')[0], 
                    (r.total_params_millions, r.star_mAP),
                    textcoords="offset points", xytext=(5, 5), fontsize=7)
    
    ax4.set_xlabel('Parameters (millions)')
    ax4.set_ylabel('mAP')
    ax4.set_title('Model Size vs Performance', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Create legend combining architecture colors and strategy markers
    arch_handles = [mpatches.Patch(color=c, label=a) for a, c in ARCH_COLORS.items() 
                   if a in [r.architecture for r in results]]
    strategy_handles = [plt.Line2D([0], [0], marker=m, color='gray', linestyle='', 
                                   markersize=8, label=s.replace('_', ' ').title())
                       for s, m in STRATEGY_MARKERS.items() if s in strategies]
    ax4.legend(handles=arch_handles + strategy_handles, loc='lower right', fontsize=7)
    
    # --- Plot 5: Top N Multi-Metric Comparison ---
    ax5 = fig.add_subplot(3, 2, 5)
    
    top_n = min(5, len(sorted_results))
    top_results = sorted_results[:top_n]
    
    x = np.arange(top_n)
    width = 0.25
    
    metrics = [
        ('mAP', [r.star_mAP for r in top_results], '#2E86AB'),
        ('Rank-1', [r.star_rank1 for r in top_results], '#A23B72'),
        ('Rank-5', [r.star_rank5 for r in top_results], '#F18F01'),
    ]
    
    for i, (name, values, color) in enumerate(metrics):
        offset = (i - 1) * width
        bars = ax5.bar(x + offset, values, width, label=name, color=color, edgecolor='white')
        
        for bar, val in zip(bars, values):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=7)
    
    ax5.set_xlabel('Experiment')
    ax5.set_ylabel('Metric Value')
    ax5.set_title(f'Top {top_n} Experiments: Multi-Metric Comparison', fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels([r.name.replace('_star_only', '').replace('_triplet', '_T').replace('_circle', '_C') 
                         for r in top_results], fontsize=8, rotation=15, ha='right')
    ax5.legend()
    ax5.set_ylim(0, 1.1)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # --- Plot 6: Training Time Comparison ---
    ax6 = fig.add_subplot(3, 2, 6)
    
    # Sort by time
    time_sorted = sorted(results, key=lambda x: x.total_time_seconds)
    names = [r.name.replace('_star_only', '').replace('_triplet', '_T').replace('_circle', '_C') 
             for r in time_sorted]
    times = [r.total_time_seconds / 60 for r in time_sorted]  # Convert to minutes
    colors = [ARCH_COLORS.get(r.architecture, '#888888') for r in time_sorted]
    
    y_pos = np.arange(len(names))
    bars = ax6.barh(y_pos, times, color=colors, edgecolor='white', linewidth=0.5)
    
    ax6.set_yticks(y_pos)
    ax6.set_yticklabels(names, fontsize=8)
    ax6.set_xlabel('Training Time (minutes)')
    ax6.set_title('Training Time by Experiment', fontweight='bold')
    
    # Add mAP annotations
    for i, (bar, r) in enumerate(zip(bars, time_sorted)):
        time_val = r.total_time_seconds / 60
        ax6.text(time_val + 0.5, bar.get_y() + bar.get_height()/2,
                f'mAP={r.star_mAP:.3f}', va='center', fontsize=7, alpha=0.8)
    
    ax6.grid(True, alpha=0.3, axis='x')
    
    # Overall title and layout
    fig.suptitle('Grid Search Summary', fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"📊 Grid search summary saved to: {output_path}")
    return output_path


# =============================================================================
# Testing / Verification
# =============================================================================

def create_mock_history() -> Dict[str, Any]:
    """Create mock training history for testing."""
    epochs = list(range(1, 31))
    
    # Simulate training loss decreasing
    train_loss = [0.9 - 0.02*e + 0.05*np.random.randn() for e in epochs]
    train_loss = [max(0.1, l) for l in train_loss]  # Clamp
    
    # Simulate triplet loss
    train_triplet = [l * 1.0 for l in train_loss]
    
    # Learning rate: warmup then cosine decay
    warmup_epochs = 3
    lr = []
    base_lr = 2e-4
    for e in epochs:
        if e <= warmup_epochs:
            lr.append(base_lr * e / warmup_epochs)
        else:
            progress = (e - warmup_epochs) / (len(epochs) - warmup_epochs)
            lr.append(base_lr * 0.5 * (1 + np.cos(np.pi * progress)))
    
    # Validation every 5 epochs
    val_map = [None] * len(epochs)
    val_rank1 = [None] * len(epochs)
    val_rank5 = [None] * len(epochs)
    
    for i, e in enumerate(epochs):
        if e % 5 == 0 or e == len(epochs):
            # Simulate improving then plateauing
            progress = e / len(epochs)
            val_map[i] = 0.3 + 0.4 * (1 - np.exp(-3*progress)) + 0.03*np.random.randn()
            val_rank1[i] = val_map[i] * 0.9 + 0.05*np.random.randn()
            val_rank5[i] = val_map[i] * 1.1 + 0.03*np.random.randn()
            val_map[i] = min(1.0, max(0, val_map[i]))
            val_rank1[i] = min(1.0, max(0, val_rank1[i]))
            val_rank5[i] = min(1.0, max(0, val_rank5[i]))
    
    return {
        'format': 'column',
        'num_epochs': len(epochs),
        'columns': {
            'epoch': epochs,
            'epoch_time': [45.0 + 5*np.random.randn() for _ in epochs],
            'learning_rate': lr,
            'train_total': train_loss,
            'train_triplet': train_triplet,
            'val_mAP': val_map,
            'val_Rank-1': val_rank1,
            'val_Rank-5': val_rank5,
        }
    }


def create_mock_results() -> List[ExperimentResult]:
    """Create mock grid search results for testing."""
    results = []
    
    configs = [
        ('swinv2-tiny', 'star_only', 'triplet', 29.0),
        ('swinv2-tiny', 'star_only', 'circle', 29.0),
        ('densenet121', 'star_only', 'triplet', 8.7),
        ('densenet121', 'star_only', 'circle', 8.7),
        ('densenet169', 'star_only', 'triplet', 14.0),
        ('densenet169', 'star_only', 'circle', 14.0),
        ('resnet50', 'star_only', 'triplet', 25.0),
        ('resnet50', 'star_only', 'circle', 25.0),
        ('convnext-tiny', 'star_only', 'triplet', 28.0),
        ('convnext-tiny', 'star_only', 'circle', 28.0),
    ]
    
    for arch, strategy, loss, params in configs:
        # Generate plausible metrics with some variance
        base_map = 0.5 + 0.2 * np.random.randn()
        base_map = min(0.85, max(0.35, base_map))
        
        results.append(ExperimentResult(
            name=f'{arch}_{strategy}_{loss}',
            architecture=arch,
            strategy=strategy,
            loss_type=loss,
            total_params_millions=params,
            batch_size=160,
            star_mAP=base_map,
            star_rank1=base_map * 0.95 + 0.02*np.random.randn(),
            star_rank5=min(1.0, base_map * 1.15 + 0.02*np.random.randn()),
            total_time_seconds=1800 + 600*np.random.randn(),
            best_epoch=25 + int(5*np.random.randn()),
        ))
    
    return results


def test_plotting(output_dir: Path = Path('./test_plots')):
    """Test all plotting functions with mock data."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Testing plotting functions...")
    
    # Test 1: Per-run training history plot
    print("\n1. Testing plot_training_history()...")
    mock_history = create_mock_history()
    history_path = output_dir / 'mock_training_history.json'
    with open(history_path, 'w') as f:
        json.dump(mock_history, f, indent=2)
    
    plot_path = plot_training_history(history_path, experiment_name='test_swinv2_star_only_triplet')
    print(f"   ✅ Created: {plot_path}")
    
    # Test 2: Grid search summary plot
    print("\n2. Testing plot_grid_search_summary()...")
    mock_results = create_mock_results()
    summary_path = plot_grid_search_summary(mock_results, output_dir)
    print(f"   ✅ Created: {summary_path}")
    
    print(f"\n✅ All tests passed! Check output in: {output_dir}")
    return True


if __name__ == '__main__':
    # Run tests when executed directly
    test_plotting()


