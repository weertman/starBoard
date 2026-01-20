"""
Generate interpretability figures for verification model.

Creates publication-quality visualizations showing:
- GradCAM heatmaps for both images in verification pairs
- Cross-attention maps showing what regions are compared
- Summary grids organized by prediction outcome (TP, TN, FP, FN)

Usage:
    python -m megastar_identity_verification.scripts.generate_interpretability_figures \
        --checkpoint path/to/best.pth \
        --output-dir interpretability_figures/ \
        --num-examples 20
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from megastar_identity_verification.config import VerificationConfig
from megastar_identity_verification.dataset import StarDatasetPairDataset
from megastar_identity_verification.interpretability import (
    VerificationGradCAM,
    AttentionExtractor,
    aggregate_attention_map,
    reshape_attention_to_spatial,
)
from megastar_identity_verification.model import VerificationModel
from megastar_identity_verification.transforms import get_test_transforms
from megastar_identity_verification.visualize import (
    create_interpretation_figure,
    create_summary_grid,
    denormalize_image,
    overlay_heatmap,
    plot_attention_grid,
    plot_attention_map,
    plot_gradcam_pair,
)


def load_model(
    checkpoint_path: str,
    device: torch.device,
) -> Tuple[VerificationModel, VerificationConfig]:
    """
    Load verification model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        
    Returns:
        (model, config) tuple
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Load config - check if it's a model config or training config
    if 'config' in checkpoint:
        config_dict = checkpoint['config']
        # Check if this is a model config (has backbone key) or training config
        if 'backbone' in config_dict:
            # VerificationConfig uses __post_init__ to convert dicts to nested dataclasses
            config = VerificationConfig(**config_dict)
        else:
            # Training config - use defaults for model config
            print("Note: Checkpoint has training config, using default model config")
            config = VerificationConfig()
    else:
        # Use default config
        print("Warning: No config in checkpoint, using defaults")
        config = VerificationConfig()
    
    # Create and load model
    model = VerificationModel(config)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model, config


def collect_examples(
    model: VerificationModel,
    dataset: StarDatasetPairDataset,
    device: torch.device,
    num_examples: int = 50,
    batch_size: int = 8,
) -> Dict[str, List[Dict]]:
    """
    Collect examples for each prediction category.
    
    Args:
        model: Verification model
        dataset: Test dataset
        device: Device to run inference on
        num_examples: Target number of examples per category
        batch_size: Batch size for inference
        
    Returns:
        Dict mapping category ('TP', 'TN', 'FP', 'FN') to list of examples
    """
    from torch.utils.data import DataLoader
    
    # Create dataloader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle to get diverse examples
        num_workers=0,  # Avoid multiprocessing issues
    )
    
    # Collect examples by category
    examples = defaultdict(list)
    target_per_category = num_examples
    
    print(f"Collecting up to {target_per_category} examples per category...")
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Scanning dataset"):
            img_a = batch['image_a'].to(device)
            img_b = batch['image_b'].to(device)
            labels = batch['label'].numpy()
            
            # Get predictions
            logits = model(img_a, img_b)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            preds = (probs > 0.5).astype(int)
            
            # Categorize each example
            for i in range(len(labels)):
                label = int(labels[i])
                pred = preds[i]
                prob = probs[i]
                
                # Determine category
                if label == 1 and pred == 1:
                    category = 'TP'
                elif label == 0 and pred == 0:
                    category = 'TN'
                elif label == 0 and pred == 1:
                    category = 'FP'
                else:  # label == 1 and pred == 0
                    category = 'FN'
                
                # Check if we need more of this category
                if len(examples[category]) < target_per_category:
                    examples[category].append({
                        'img_a': img_a[i].detach().cpu(),
                        'img_b': img_b[i].detach().cpu(),
                        'label': label,
                        'prediction': prob,
                        'category': category,
                    })
            
            # Check if we have enough
            if all(len(examples[cat]) >= target_per_category 
                   for cat in ['TP', 'TN', 'FP', 'FN']):
                break
    
    # Report counts
    print("\nCollected examples:")
    for cat in ['TP', 'TN', 'FP', 'FN']:
        print(f"  {cat}: {len(examples[cat])}")
    
    return dict(examples)


def generate_gradcam_figures(
    model: VerificationModel,
    examples: Dict[str, List[Dict]],
    output_dir: Path,
    device: torch.device,
    max_per_category: int = 5,
):
    """
    Generate GradCAM figures for examples.
    
    Args:
        model: Verification model
        examples: Dict of examples by category
        output_dir: Output directory
        device: Device to run inference on
        max_per_category: Max figures per category
    """
    gradcam_dir = output_dir / 'gradcam'
    gradcam_dir.mkdir(exist_ok=True)
    
    # Initialize GradCAM
    gradcam = VerificationGradCAM(model)
    
    try:
        for category, category_examples in examples.items():
            print(f"\nGenerating GradCAM for {category}...")
            
            for idx, example in enumerate(tqdm(category_examples[:max_per_category])):
                img_a = example['img_a'].unsqueeze(0).to(device)
                img_b = example['img_b'].unsqueeze(0).to(device)
                
                # Compute GradCAM
                try:
                    heatmap_a, heatmap_b = gradcam(img_a, img_b)
                except Exception as e:
                    print(f"  Warning: GradCAM failed for {category}_{idx}: {e}")
                    continue
                
                # Create figure
                save_path = gradcam_dir / f'{category.lower()}_{idx:03d}.pdf'
                plot_gradcam_pair(
                    img_a=example['img_a'],
                    img_b=example['img_b'],
                    heatmap_a=heatmap_a,
                    heatmap_b=heatmap_b,
                    prediction=example['prediction'],
                    label=example['label'],
                    title=f"{category}: Pred={example['prediction']:.1%}, True={'Same' if example['label'] else 'Diff'}",
                    save_path=save_path,
                )
    finally:
        gradcam.remove_hooks()
    
    print(f"  Saved to {gradcam_dir}")


def generate_attention_figures(
    model: VerificationModel,
    examples: Dict[str, List[Dict]],
    output_dir: Path,
    device: torch.device,
    max_per_category: int = 3,
):
    """
    Generate attention map figures for examples.
    
    Args:
        model: Verification model
        examples: Dict of examples by category
        output_dir: Output directory
        device: Device to run inference on
        max_per_category: Max figures per category
    """
    attention_dir = output_dir / 'attention'
    attention_dir.mkdir(exist_ok=True)
    
    # Initialize attention extractor
    extractor = AttentionExtractor(model)
    
    try:
        for category, category_examples in examples.items():
            print(f"\nGenerating attention maps for {category}...")
            
            for idx, example in enumerate(tqdm(category_examples[:max_per_category])):
                img_a = example['img_a'].unsqueeze(0).to(device)
                img_b = example['img_b'].unsqueeze(0).to(device)
                
                # Extract attention
                try:
                    attn_a2b, attn_b2a = extractor.get_cross_attention_maps(img_a, img_b)
                except Exception as e:
                    print(f"  Warning: Attention extraction failed for {category}_{idx}: {e}")
                    continue
                
                # Aggregated attention map
                save_path = attention_dir / f'{category.lower()}_{idx:03d}_aggregated.pdf'
                plot_attention_map(
                    img_a=example['img_a'],
                    img_b=example['img_b'],
                    attention_weights=attn_a2b,
                    source_position=None,  # Aggregated
                    title=f"{category}: Aggregated Attention A→B",
                    save_path=save_path,
                )
                
                # Grid of attention from multiple positions
                save_path = attention_dir / f'{category.lower()}_{idx:03d}_grid.pdf'
                plot_attention_grid(
                    img_a=example['img_a'],
                    img_b=example['img_b'],
                    attention_weights=attn_a2b,
                    title=f"{category}: Attention from Multiple Positions",
                    save_path=save_path,
                )
    finally:
        extractor.remove_hooks()
    
    print(f"  Saved to {attention_dir}")


def generate_combined_figures(
    model: VerificationModel,
    examples: Dict[str, List[Dict]],
    output_dir: Path,
    device: torch.device,
    max_per_category: int = 3,
):
    """
    Generate combined GradCAM + attention figures.
    
    Args:
        model: Verification model
        examples: Dict of examples by category
        output_dir: Output directory
        device: Device to run inference on
        max_per_category: Max figures per category
    """
    combined_dir = output_dir / 'combined'
    combined_dir.mkdir(exist_ok=True)
    
    for category, category_examples in examples.items():
        print(f"\nGenerating combined figures for {category}...")
        
        for idx, example in enumerate(tqdm(category_examples[:max_per_category])):
            save_path = combined_dir / f'{category.lower()}_{idx:03d}_full.pdf'
            
            try:
                create_interpretation_figure(
                    model=model,
                    img_a=example['img_a'],
                    img_b=example['img_b'],
                    label=example['label'],
                    device=device,
                    save_path=save_path,
                )
            except Exception as e:
                print(f"  Warning: Combined figure failed for {category}_{idx}: {e}")
                continue
    
    print(f"  Saved to {combined_dir}")


def generate_summary_figure(
    model: VerificationModel,
    examples: Dict[str, List[Dict]],
    output_dir: Path,
    device: torch.device,
    examples_per_category: int = 4,
):
    """
    Generate summary grid figure for publication.
    
    Args:
        model: Verification model
        examples: Dict of examples by category
        output_dir: Output directory
        device: Device to run inference on
        examples_per_category: Examples per category in grid
    """
    print("\nGenerating summary figure...")
    
    # Initialize GradCAM
    gradcam = VerificationGradCAM(model)
    
    try:
        # Collect examples with heatmaps
        summary_examples = []
        
        for category in ['TP', 'FP', 'FN', 'TN']:  # Order for visual clarity
            for example in examples.get(category, [])[:examples_per_category]:
                img_a = example['img_a'].unsqueeze(0).to(device)
                img_b = example['img_b'].unsqueeze(0).to(device)
                
                try:
                    heatmap_a, heatmap_b = gradcam(img_a, img_b)
                except Exception:
                    continue
                
                summary_examples.append({
                    'img_a': denormalize_image(example['img_a']),
                    'img_b': denormalize_image(example['img_b']),
                    'heatmap_a': heatmap_a[0],
                    'heatmap_b': heatmap_b[0],
                    'prediction': example['prediction'],
                    'label': example['label'],
                    'category': category,
                })
    finally:
        gradcam.remove_hooks()
    
    if summary_examples:
        save_path = output_dir / 'summary_figure.pdf'
        create_summary_grid(
            examples=summary_examples,
            n_cols=4,
            save_path=save_path,
        )
        print(f"  Saved to {save_path}")
    else:
        print("  Warning: No examples collected for summary figure")


def main():
    parser = argparse.ArgumentParser(
        description='Generate interpretability figures for verification model'
    )
    parser.add_argument(
        '--checkpoint', '-c',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='verification_interpretability_figures',
        help='Output directory for figures'
    )
    parser.add_argument(
        '--data-root',
        type=str,
        default='D:/star_identification/star_dataset',
        help='Path to star_dataset'
    )
    parser.add_argument(
        '--num-examples', '-n',
        type=int,
        default=20,
        help='Number of examples to collect per category'
    )
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=8,
        help='Batch size for inference'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use (cuda or cpu)'
    )
    parser.add_argument(
        '--skip-gradcam',
        action='store_true',
        help='Skip GradCAM figure generation'
    )
    parser.add_argument(
        '--skip-attention',
        action='store_true',
        help='Skip attention figure generation'
    )
    parser.add_argument(
        '--skip-combined',
        action='store_true',
        help='Skip combined figure generation'
    )
    parser.add_argument(
        '--skip-summary',
        action='store_true',
        help='Skip summary figure generation'
    )
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    model, config = load_model(args.checkpoint, device)
    print(f"  Model loaded successfully")
    
    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
    
    # Create test dataset
    print(f"\nLoading test dataset from {args.data_root}...")
    transform = get_test_transforms(config.image_size)
    
    dataset = StarDatasetPairDataset.from_star_dataset(
        data_root=args.data_root,
        mode='test',
        transform=transform,
        pairs_per_epoch=args.num_examples * 20,  # Sample more to ensure we get enough of each category
        positive_ratio=0.5,
        seed=42,
    )
    
    # Collect examples
    examples = collect_examples(
        model=model,
        dataset=dataset,
        device=device,
        num_examples=args.num_examples,
        batch_size=args.batch_size,
    )
    
    # Generate figures
    if not args.skip_gradcam:
        generate_gradcam_figures(
            model=model,
            examples=examples,
            output_dir=output_dir,
            device=device,
        )
    
    if not args.skip_attention:
        generate_attention_figures(
            model=model,
            examples=examples,
            output_dir=output_dir,
            device=device,
        )
    
    if not args.skip_combined:
        generate_combined_figures(
            model=model,
            examples=examples,
            output_dir=output_dir,
            device=device,
        )
    
    if not args.skip_summary:
        generate_summary_figure(
            model=model,
            examples=examples,
            output_dir=output_dir,
            device=device,
        )
    
    # Save metadata
    metadata = {
        'checkpoint': str(args.checkpoint),
        'data_root': str(args.data_root),
        'num_examples_requested': args.num_examples,
        'examples_collected': {cat: len(exs) for cat, exs in examples.items()},
        'timestamp': timestamp,
    }
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Interpretability figures generated successfully!")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

