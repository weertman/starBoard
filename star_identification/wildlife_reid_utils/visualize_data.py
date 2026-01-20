#!/usr/bin/env python
"""
Visualize Training Data for MegaStarID.

Shows what the input data looks like before and after augmentation.
Creates visual grids of samples from:
- Wildlife10k (diverse species)
- star_dataset (sea stars)

Saves to wildlife_reid_utils/visualizations/ by default.

Usage:
    python -m wildlife_reid_utils.visualize_data
    python -m wildlife_reid_utils.visualize_data --samples 8 --augmentations 4
    python -m wildlife_reid_utils.visualize_data --dataset star
    python -m wildlife_reid_utils.visualize_data --show  # Display instead of save
    python -m wildlife_reid_utils.visualize_data --output custom_path.png
"""
import argparse
import random
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from megastarid.transforms import (
    get_wildlife_train_transforms,
    get_wildlife_test_transforms,
    get_star_train_transforms,
    get_star_test_transforms,
)


# =============================================================================
# Data Loading Utilities
# =============================================================================

def load_star_samples(
    data_root: str = './star_dataset',
    num_samples: int = 8,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Load random samples from star_dataset.
    
    Returns list of dicts with 'image' (PIL), 'identity', 'path', 'folder'.
    """
    data_root = Path(data_root)
    
    # Load metadata
    metadata_path = data_root / 'metadata_temporal.csv'
    if not metadata_path.exists():
        metadata_path = data_root / 'metadata.csv'
    
    if not metadata_path.exists():
        print(f"Error: Metadata not found at {data_root}")
        return []
    
    df = pd.read_csv(metadata_path)
    
    # Sample random rows
    random.seed(seed)
    if len(df) > num_samples:
        indices = random.sample(range(len(df)), num_samples)
    else:
        indices = list(range(len(df)))
    
    samples = []
    for idx in indices:
        row = df.iloc[idx]
        
        # Build path
        if 'path' in row and pd.notna(row['path']):
            path = row['path']
            if not Path(path).is_absolute():
                path = data_root / path
        else:
            path = data_root / row['folder'] / row['filename']
        
        try:
            image = Image.open(path).convert('RGB')
            samples.append({
                'image': image,
                'identity': row['identity'],
                'path': str(path),
                'folder': row.get('folder', 'unknown'),
                'dataset': 'star_dataset',
                'species': 'sea_star',
            })
        except Exception as e:
            print(f"Warning: Could not load {path}: {e}")
    
    return samples


def load_star_samples_stratified_by_folder(
    data_root: str = './star_dataset',
    num_samples: int = 8,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Load samples from star_dataset, stratified by data source.
    
    Path structure: star_dataset/{datasource}/{identity}/{outing}/{filename}
    
    Randomly selects N data sources, then picks 1 random identity from each,
    then picks 1 random image from that identity.
    
    This ensures diversity across data sources (ADULT_FHL_STARS, PWS_2023, etc.)
    rather than oversampling from larger sources.
    
    Returns list of dicts with 'image' (PIL), 'identity', 'path', 'datasource'.
    """
    data_root = Path(data_root)
    
    # Load metadata
    metadata_path = data_root / 'metadata_temporal.csv'
    if not metadata_path.exists():
        metadata_path = data_root / 'metadata.csv'
    
    if not metadata_path.exists():
        print(f"Error: Metadata not found at {data_root}")
        return []
    
    df = pd.read_csv(metadata_path)
    
    # Extract datasource from path
    # Path format: .../star_dataset/{datasource}/{identity}/{outing}/{filename}
    def extract_datasource(path_str):
        path = Path(path_str)
        # Find parts after star_dataset (or star_dataset_resized)
        parts = path.parts
        for i, part in enumerate(parts):
            if 'star_dataset' in part.lower():
                if i + 1 < len(parts):
                    return parts[i + 1]
        # Fallback: return first directory component of relative path
        return parts[0] if parts else 'unknown'
    
    df['datasource'] = df['path'].apply(extract_datasource)
    
    # Get unique datasources
    datasources = df['datasource'].unique().tolist()
    
    random.seed(seed)
    
    # Sample N datasources (or all if fewer available)
    num_sources_to_sample = min(num_samples, len(datasources))
    sampled_sources = random.sample(datasources, num_sources_to_sample)
    
    print(f"  Stratified sampling: {num_sources_to_sample} data sources selected")
    print(f"  Sources: {sampled_sources}")
    
    samples = []
    for datasource in sampled_sources:
        # Get all rows from this datasource
        source_df = df[df['datasource'] == datasource]
        
        # Get unique identities in this datasource
        identities = source_df['identity'].unique().tolist()
        
        # Pick 1 random identity
        identity = random.choice(identities)
        
        # Get all rows for this identity in this datasource
        identity_df = source_df[source_df['identity'] == identity]
        
        # Pick 1 random image
        row = identity_df.sample(n=1, random_state=random.randint(0, 10000)).iloc[0]
        
        # Get path
        path = row['path']
        if not Path(path).is_absolute():
            path = data_root / path
        
        try:
            image = Image.open(path).convert('RGB')
            samples.append({
                'image': image,
                'identity': row['identity'],
                'path': str(path),
                'folder': datasource,  # Use datasource as folder for display
                'dataset': 'star_dataset',
                'species': 'sea_star',
            })
        except Exception as e:
            print(f"Warning: Could not load {path}: {e}")
    
    return samples


def load_wildlife_samples(
    data_root: str = './wildlifeReID',
    num_samples: int = 8,
    seed: int = 42,
    exclude_datasets: List[str] = None,
) -> List[Dict[str, Any]]:
    """
    Load random samples from Wildlife10k.
    
    Returns list of dicts with 'image' (PIL), 'identity', 'path', 'dataset', 'species'.
    """
    data_root = Path(data_root)
    
    if exclude_datasets is None:
        exclude_datasets = ['SeaStarReID2023']
    
    # Load metadata
    metadata_path = data_root / 'metadata.csv'
    if not metadata_path.exists():
        print(f"Error: Metadata not found at {data_root}")
        return []
    
    df = pd.read_csv(metadata_path)
    
    # Filter out excluded datasets
    df = df[~df['dataset'].isin(exclude_datasets)]
    
    # Try to get diverse samples (one per dataset if possible)
    random.seed(seed)
    datasets = df['dataset'].unique().tolist()
    random.shuffle(datasets)
    
    samples = []
    datasets_used = []
    
    # First, try one sample per dataset
    for ds in datasets:
        if len(samples) >= num_samples:
            break
        
        ds_df = df[df['dataset'] == ds]
        if len(ds_df) > 0:
            row = ds_df.sample(1, random_state=seed).iloc[0]
            path = data_root / row['path']
            
            try:
                image = Image.open(path).convert('RGB')
                samples.append({
                    'image': image,
                    'identity': row['identity'],
                    'path': str(path),
                    'dataset': row['dataset'],
                    'species': row.get('species', 'unknown'),
                })
                datasets_used.append(ds)
            except Exception as e:
                print(f"Warning: Could not load {path}: {e}")
    
    # Fill remaining with random samples
    if len(samples) < num_samples:
        remaining = num_samples - len(samples)
        remaining_df = df[~df['dataset'].isin(datasets_used)]
        if len(remaining_df) > 0:
            sample_rows = remaining_df.sample(min(remaining, len(remaining_df)), random_state=seed)
            
            for _, row in sample_rows.iterrows():
                path = data_root / row['path']
                try:
                    image = Image.open(path).convert('RGB')
                    samples.append({
                        'image': image,
                        'identity': row['identity'],
                        'path': str(path),
                        'dataset': row['dataset'],
                        'species': row.get('species', 'unknown'),
                    })
                except:
                    pass
    
    return samples


# =============================================================================
# Tensor to Image Conversion
# =============================================================================

def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """
    Convert a normalized tensor back to PIL Image.
    
    Assumes ImageNet normalization.
    """
    # Denormalize
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    tensor = tensor.clone()
    tensor = tensor * std + mean
    tensor = tensor.clamp(0, 1)
    
    # Convert to numpy
    np_img = tensor.permute(1, 2, 0).numpy()
    np_img = (np_img * 255).astype(np.uint8)
    
    return Image.fromarray(np_img)


# =============================================================================
# Visualization Functions
# =============================================================================

def visualize_samples_with_augmentations(
    samples: List[Dict[str, Any]],
    train_transform,
    test_transform,
    num_augmentations: int = 4,
    image_size: int = 384,
    title: str = "Data Visualization",
    figsize_per_image: float = 2.0,
) -> plt.Figure:
    """
    Create a visualization grid showing original and augmented images.
    
    Layout:
        Each row is one sample.
        Column 0: Original image
        Column 1: Test transform (no augmentation)
        Columns 2+: Multiple augmentation variations
    """
    num_samples = len(samples)
    num_cols = 2 + num_augmentations  # Original + Test + Augmentations
    
    fig_width = num_cols * figsize_per_image
    fig_height = num_samples * figsize_per_image + 1.5  # Extra space for title
    
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    # Title
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    
    # Create grid
    gs = gridspec.GridSpec(num_samples, num_cols, figure=fig,
                           wspace=0.05, hspace=0.15,
                           top=0.92, bottom=0.02, left=0.02, right=0.98)
    
    # Column headers
    col_headers = ['Original', 'Test (no aug)'] + [f'Aug {i+1}' for i in range(num_augmentations)]
    
    for sample_idx, sample in enumerate(samples):
        original_image = sample['image']
        identity = sample.get('identity', 'unknown')
        dataset = sample.get('dataset', 'unknown')
        species = sample.get('species', '')
        
        # Row label
        if dataset == 'star_dataset':
            row_label = f"Star: {identity}"
        else:
            ds_short = dataset[:15] + '...' if len(dataset) > 15 else dataset
            row_label = f"{ds_short}"
        
        for col_idx in range(num_cols):
            ax = fig.add_subplot(gs[sample_idx, col_idx])
            
            if col_idx == 0:
                # Original image (resized for display)
                display_img = original_image.copy()
                display_img = display_img.resize((image_size, image_size), Image.Resampling.LANCZOS)
                ax.imshow(display_img)
                ax.set_ylabel(row_label, fontsize=8, rotation=0, ha='right', va='center')
            elif col_idx == 1:
                # Test transform (no augmentation)
                transformed = test_transform(original_image.copy())
                display_img = tensor_to_pil(transformed)
                ax.imshow(display_img)
            else:
                # Training augmentation
                transformed = train_transform(original_image.copy())
                display_img = tensor_to_pil(transformed)
                ax.imshow(display_img)
            
            # Column headers (first row only)
            if sample_idx == 0:
                ax.set_title(col_headers[col_idx], fontsize=9, pad=3)
            
            ax.axis('off')
    
    return fig


def visualize_dataset_comparison(
    star_samples: List[Dict[str, Any]],
    wildlife_samples: List[Dict[str, Any]],
    star_train_transform,
    wildlife_train_transform,
    test_transform,
    num_augmentations: int = 3,
    image_size: int = 384,
    figsize_per_image: float = 1.8,
) -> plt.Figure:
    """
    Create a side-by-side comparison of star_dataset and Wildlife10k augmentations.
    
    Shows both datasets with their respective augmentation strategies.
    """
    num_star = len(star_samples)
    num_wildlife = len(wildlife_samples)
    num_cols = 1 + 1 + num_augmentations  # Original + Test + Augmentations
    
    total_rows = num_star + num_wildlife + 2  # +2 for section headers
    
    fig_width = num_cols * figsize_per_image
    fig_height = total_rows * figsize_per_image * 0.9
    
    fig = plt.figure(figsize=(fig_width, fig_height))
    fig.suptitle("Dataset Augmentation Comparison", fontsize=14, fontweight='bold', y=0.995)
    
    # Create grid with gaps for headers
    gs = gridspec.GridSpec(total_rows, num_cols, figure=fig,
                           wspace=0.05, hspace=0.12,
                           top=0.96, bottom=0.01, left=0.12, right=0.98)
    
    col_headers = ['Original', 'Test'] + [f'Aug {i+1}' for i in range(num_augmentations)]
    
    current_row = 0
    
    # === STAR DATASET SECTION ===
    # Section header
    ax_header = fig.add_subplot(gs[current_row, :])
    ax_header.text(0.5, 0.5, '🌟 star_dataset (Albumentations - Underwater Optimized)',
                   transform=ax_header.transAxes, ha='center', va='center',
                   fontsize=11, fontweight='bold', 
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    ax_header.axis('off')
    current_row += 1
    
    for sample_idx, sample in enumerate(star_samples):
        original_image = sample['image']
        identity = sample.get('identity', 'unknown')
        
        for col_idx in range(num_cols):
            ax = fig.add_subplot(gs[current_row, col_idx])
            
            if col_idx == 0:
                display_img = original_image.copy()
                display_img = display_img.resize((image_size, image_size), Image.Resampling.LANCZOS)
                ax.imshow(display_img)
                ax.set_ylabel(f"{identity[:12]}", fontsize=7, rotation=0, ha='right', va='center')
            elif col_idx == 1:
                transformed = test_transform(original_image.copy())
                display_img = tensor_to_pil(transformed)
                ax.imshow(display_img)
            else:
                transformed = star_train_transform(original_image.copy())
                display_img = tensor_to_pil(transformed)
                ax.imshow(display_img)
            
            if sample_idx == 0:
                ax.set_title(col_headers[col_idx], fontsize=8, pad=2)
            
            ax.axis('off')
        
        current_row += 1
    
    # === WILDLIFE10K SECTION ===
    # Section header
    ax_header = fig.add_subplot(gs[current_row, :])
    ax_header.text(0.5, 0.5, '🦁 Wildlife10k (Torchvision - Standard)',
                   transform=ax_header.transAxes, ha='center', va='center',
                   fontsize=11, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    ax_header.axis('off')
    current_row += 1
    
    for sample_idx, sample in enumerate(wildlife_samples):
        original_image = sample['image']
        dataset = sample.get('dataset', 'unknown')
        
        for col_idx in range(num_cols):
            ax = fig.add_subplot(gs[current_row, col_idx])
            
            if col_idx == 0:
                display_img = original_image.copy()
                display_img = display_img.resize((image_size, image_size), Image.Resampling.LANCZOS)
                ax.imshow(display_img)
                ds_short = dataset[:12] + '..' if len(dataset) > 12 else dataset
                ax.set_ylabel(ds_short, fontsize=6, rotation=0, ha='right', va='center')
            elif col_idx == 1:
                transformed = test_transform(original_image.copy())
                display_img = tensor_to_pil(transformed)
                ax.imshow(display_img)
            else:
                transformed = wildlife_train_transform(original_image.copy())
                display_img = tensor_to_pil(transformed)
                ax.imshow(display_img)
            
            ax.axis('off')
        
        current_row += 1
    
    return fig


def visualize_augmentation_variety(
    sample: Dict[str, Any],
    train_transform,
    num_variations: int = 16,
    image_size: int = 384,
    cols: int = 4,
) -> plt.Figure:
    """
    Show many augmentation variations for a single image.
    
    Useful for understanding augmentation diversity.
    """
    rows = (num_variations + 1 + cols - 1) // cols  # +1 for original
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    fig.suptitle(f"Augmentation Variations\n{sample.get('identity', 'Unknown')}", 
                 fontsize=12, fontweight='bold')
    
    axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    
    original_image = sample['image']
    
    # First image is original
    display_img = original_image.copy().resize((image_size, image_size), Image.Resampling.LANCZOS)
    axes[0].imshow(display_img)
    axes[0].set_title('Original', fontsize=9)
    axes[0].axis('off')
    
    # Rest are augmented
    for i in range(1, min(num_variations + 1, len(axes))):
        transformed = train_transform(original_image.copy())
        display_img = tensor_to_pil(transformed)
        axes[i].imshow(display_img)
        axes[i].set_title(f'Aug {i}', fontsize=8)
        axes[i].axis('off')
    
    # Hide unused axes
    for i in range(num_variations + 1, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Visualize training data and augmentations')
    
    parser.add_argument('--dataset', type=str, default='both',
                        choices=['star', 'wildlife', 'both'],
                        help='Which dataset to visualize')
    parser.add_argument('--samples', type=int, default=6,
                        help='Number of samples per dataset')
    parser.add_argument('--augmentations', type=int, default=4,
                        help='Number of augmentation variations to show')
    parser.add_argument('--image-size', type=int, default=384,
                        help='Image size for transforms')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for sampling')
    parser.add_argument('--output', type=str, default='auto',
                        help='Output file path (default: auto-save to visualizations/)')
    parser.add_argument('--show', action='store_true',
                        help='Display instead of saving')
    parser.add_argument('--variety', action='store_true',
                        help='Show many variations for one image')
    parser.add_argument('--variety-count', type=int, default=16,
                        help='Number of variations for --variety mode')
    parser.add_argument('--stratify-by-folder', action='store_true',
                        help='Sample one identity per folder for diversity across observation sessions')
    
    # Paths
    parser.add_argument('--star-root', type=str, default='./star_dataset',
                        help='Path to star_dataset')
    parser.add_argument('--wildlife-root', type=str, default='./wildlifeReID',
                        help='Path to Wildlife10k')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("DATA VISUALIZATION")
    print("=" * 60)
    
    # Get transforms
    print(f"\nImage size: {args.image_size}x{args.image_size}")
    
    star_train_transform = get_star_train_transforms(args.image_size)
    star_test_transform = get_star_test_transforms(args.image_size)
    wildlife_train_transform = get_wildlife_train_transforms(args.image_size)
    wildlife_test_transform = get_wildlife_test_transforms(args.image_size)
    
    # Load samples
    star_samples = []
    wildlife_samples = []
    
    if args.dataset in ['star', 'both']:
        if args.stratify_by_folder:
            print(f"\nLoading {args.samples} samples from star_dataset (stratified by folder)...")
            star_samples = load_star_samples_stratified_by_folder(
                args.star_root, args.samples, args.seed
            )
        else:
            print(f"\nLoading {args.samples} samples from star_dataset...")
            star_samples = load_star_samples(
                args.star_root, args.samples, args.seed
            )
        print(f"  Loaded {len(star_samples)} samples")
    
    if args.dataset in ['wildlife', 'both']:
        print(f"\nLoading {args.samples} samples from Wildlife10k...")
        wildlife_samples = load_wildlife_samples(
            args.wildlife_root, args.samples, args.seed
        )
        print(f"  Loaded {len(wildlife_samples)} samples")
    
    if not star_samples and not wildlife_samples:
        print("\nNo samples loaded! Check data paths.")
        return
    
    # Create visualization
    print(f"\nCreating visualization...")
    
    if args.variety:
        # Show many variations for one sample
        sample = star_samples[0] if star_samples else wildlife_samples[0]
        transform = star_train_transform if star_samples else wildlife_train_transform
        fig = visualize_augmentation_variety(
            sample, transform,
            num_variations=args.variety_count,
            image_size=args.image_size,
        )
    elif args.dataset == 'both' and star_samples and wildlife_samples:
        # Side-by-side comparison
        fig = visualize_dataset_comparison(
            star_samples=star_samples,
            wildlife_samples=wildlife_samples,
            star_train_transform=star_train_transform,
            wildlife_train_transform=wildlife_train_transform,
            test_transform=star_test_transform,
            num_augmentations=args.augmentations,
            image_size=args.image_size,
        )
    else:
        # Single dataset visualization
        samples = star_samples if star_samples else wildlife_samples
        train_transform = star_train_transform if star_samples else wildlife_train_transform
        test_transform = star_test_transform if star_samples else wildlife_test_transform
        title = "star_dataset" if star_samples else "Wildlife10k"
        
        fig = visualize_samples_with_augmentations(
            samples=samples,
            train_transform=train_transform,
            test_transform=test_transform,
            num_augmentations=args.augmentations,
            image_size=args.image_size,
            title=f"{title} - Original vs Augmented",
        )
    
    # Save or display
    if args.show:
        print("\nDisplaying... (close window to exit)")
        plt.show()
    else:
        # Determine output path
        if args.output == 'auto':
            # Auto-generate filename based on options
            viz_dir = Path(__file__).parent / 'visualizations'
            
            stratified_suffix = "_stratified" if args.stratify_by_folder else ""
            
            if args.variety:
                filename = f"augmentation_variety_{args.dataset}{stratified_suffix}.png"
            elif args.dataset == 'both':
                filename = f"dataset_comparison{stratified_suffix}.png"
            else:
                filename = f"{args.dataset}_augmentations{stratified_suffix}.png"
            
            output_path = viz_dir / filename
        else:
            output_path = Path(args.output)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"\nSaved to: {output_path}")
    
    plt.close(fig)
    print("\nDone!")


if __name__ == '__main__':
    main()

