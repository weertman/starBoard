"""
Visualization utilities for verification model interpretability.

Creates publication-quality figures showing GradCAM heatmaps and
cross-attention patterns.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap

# Custom colormap for attention (blue -> white -> red)
ATTENTION_CMAP = LinearSegmentedColormap.from_list(
    'attention',
    ['#2166ac', '#67a9cf', '#d1e5f0', '#f7f7f7', '#fddbc7', '#ef8a62', '#b2182b']
)

# GradCAM colormap (transparent -> blue -> green -> yellow -> red)
GRADCAM_CMAP = plt.cm.jet


def denormalize_image(
    img: Union[torch.Tensor, np.ndarray],
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
) -> np.ndarray:
    """
    Denormalize image tensor for visualization.
    
    Args:
        img: (C, H, W) or (H, W, C) normalized image
        mean: Normalization mean
        std: Normalization std
        
    Returns:
        (H, W, C) uint8 image in [0, 255]
    """
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    
    # Handle (C, H, W) format
    if img.shape[0] == 3:
        img = img.transpose(1, 2, 0)
    
    # Denormalize
    mean = np.array(mean)
    std = np.array(std)
    img = img * std + mean
    
    # Clip and convert to uint8
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    
    return img


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: plt.cm.ScalarMappable = GRADCAM_CMAP,
) -> np.ndarray:
    """
    Overlay heatmap on image.
    
    Args:
        image: (H, W, 3) RGB image in [0, 255]
        heatmap: (H_heat, W_heat) heatmap in [0, 1]
        alpha: Blend factor (0 = only image, 1 = only heatmap)
        colormap: Matplotlib colormap
        
    Returns:
        (H, W, 3) RGB image with heatmap overlay
    """
    # Resize heatmap to image size
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # Apply colormap
    heatmap_colored = colormap(heatmap_resized)[:, :, :3]  # Remove alpha
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
    
    # Blend
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
    
    return overlay


def plot_gradcam_pair(
    img_a: Union[torch.Tensor, np.ndarray],
    img_b: Union[torch.Tensor, np.ndarray],
    heatmap_a: np.ndarray,
    heatmap_b: np.ndarray,
    prediction: float,
    label: int,
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (12, 4),
) -> plt.Figure:
    """
    Create side-by-side GradCAM visualization for an image pair.
    
    Args:
        img_a: First image (normalized tensor or uint8 array)
        img_b: Second image
        heatmap_a: GradCAM heatmap for image A
        heatmap_b: GradCAM heatmap for image B
        prediction: Model prediction probability (0-1)
        label: Ground truth label (0 or 1)
        title: Optional title
        save_path: Optional path to save figure
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    # Denormalize images if needed
    if isinstance(img_a, torch.Tensor) or img_a.max() <= 1.0:
        img_a = denormalize_image(img_a)
    if isinstance(img_b, torch.Tensor) or img_b.max() <= 1.0:
        img_b = denormalize_image(img_b)
    
    # Handle batch dimension
    if heatmap_a.ndim == 3:
        heatmap_a = heatmap_a[0]
    if heatmap_b.ndim == 3:
        heatmap_b = heatmap_b[0]
    
    # Create overlays
    overlay_a = overlay_heatmap(img_a, heatmap_a)
    overlay_b = overlay_heatmap(img_b, heatmap_b)
    
    # Create figure
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    
    # Original images
    axes[0].imshow(img_a)
    axes[0].set_title('Image A', fontsize=10)
    axes[0].axis('off')
    
    axes[1].imshow(img_b)
    axes[1].set_title('Image B', fontsize=10)
    axes[1].axis('off')
    
    # GradCAM overlays
    axes[2].imshow(overlay_a)
    axes[2].set_title('GradCAM A', fontsize=10)
    axes[2].axis('off')
    
    axes[3].imshow(overlay_b)
    axes[3].set_title('GradCAM B', fontsize=10)
    axes[3].axis('off')
    
    # Add prediction info
    pred_label = "Same" if prediction > 0.5 else "Different"
    true_label = "Same" if label == 1 else "Different"
    correct = (prediction > 0.5) == (label == 1)
    color = 'green' if correct else 'red'
    
    info_text = f"Pred: {pred_label} ({prediction:.2%}) | True: {true_label}"
    fig.suptitle(
        title or info_text,
        fontsize=12,
        color=color if title is None else 'black',
    )
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    return fig


def plot_attention_map(
    img_a: Union[torch.Tensor, np.ndarray],
    img_b: Union[torch.Tensor, np.ndarray],
    attention_weights: np.ndarray,
    source_position: Optional[Tuple[int, int]] = None,
    feature_size: Tuple[int, int] = (7, 7),
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (10, 4),
) -> plt.Figure:
    """
    Visualize cross-attention from image A to image B.
    
    Args:
        img_a: Source image
        img_b: Target image
        attention_weights: (N, N) attention weights where N = H*W
        source_position: (row, col) position in source to highlight
                        If None, shows aggregated attention
        feature_size: (H, W) spatial dimensions of feature map
        title: Optional title
        save_path: Optional path to save figure
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    from .interpretability import (
        reshape_attention_to_spatial,
        aggregate_attention_map,
    )
    
    # Denormalize images
    if isinstance(img_a, torch.Tensor) or (isinstance(img_a, np.ndarray) and img_a.max() <= 1.0):
        img_a = denormalize_image(img_a)
    if isinstance(img_b, torch.Tensor) or (isinstance(img_b, np.ndarray) and img_b.max() <= 1.0):
        img_b = denormalize_image(img_b)
    
    # Handle batch dimension
    if attention_weights.ndim == 3:
        attention_weights = attention_weights[0]
    
    H, W = feature_size
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Image A with source position marker
    axes[0].imshow(img_a)
    axes[0].set_title('Source (Image A)', fontsize=10)
    axes[0].axis('off')
    
    if source_position is not None:
        # Mark source position
        row, col = source_position
        img_h, img_w = img_a.shape[:2]
        cell_h, cell_w = img_h / H, img_w / W
        
        # Draw marker at center of cell
        center_y = (row + 0.5) * cell_h
        center_x = (col + 0.5) * cell_w
        axes[0].plot(center_x, center_y, 'r*', markersize=15, markeredgecolor='white')
        
        # Get attention for this position
        spatial_attn = reshape_attention_to_spatial(
            attention_weights[np.newaxis], feature_size
        )[0]
        attn_map = spatial_attn[row, col]  # (H, W)
    else:
        # Aggregate attention
        attn_map = aggregate_attention_map(
            attention_weights[np.newaxis], feature_size, 'mean'
        )[0]
    
    # Resize attention map to image size
    attn_resized = cv2.resize(attn_map, (img_b.shape[1], img_b.shape[0]))
    
    # Normalize for visualization
    attn_resized = (attn_resized - attn_resized.min()) / (attn_resized.max() - attn_resized.min() + 1e-8)
    
    # Image B
    axes[1].imshow(img_b)
    axes[1].set_title('Target (Image B)', fontsize=10)
    axes[1].axis('off')
    
    # Attention overlay on Image B
    overlay = overlay_heatmap(img_b, attn_resized, alpha=0.6, colormap=ATTENTION_CMAP)
    axes[2].imshow(overlay)
    axes[2].set_title('Attention A→B', fontsize=10)
    axes[2].axis('off')
    
    if title:
        fig.suptitle(title, fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    return fig


def plot_attention_grid(
    img_a: Union[torch.Tensor, np.ndarray],
    img_b: Union[torch.Tensor, np.ndarray],
    attention_weights: np.ndarray,
    feature_size: Tuple[int, int] = (7, 7),
    grid_positions: Optional[List[Tuple[int, int]]] = None,
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (14, 10),
) -> plt.Figure:
    """
    Grid showing attention from multiple source positions.
    
    Args:
        img_a: Source image
        img_b: Target image
        attention_weights: (N, N) attention weights
        feature_size: (H, W) spatial dimensions
        grid_positions: List of (row, col) positions to show
                       If None, uses 3x3 grid of positions
        title: Optional title
        save_path: Optional path to save figure
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    from .interpretability import reshape_attention_to_spatial
    
    # Denormalize images
    if isinstance(img_a, torch.Tensor) or (isinstance(img_a, np.ndarray) and img_a.max() <= 1.0):
        img_a = denormalize_image(img_a)
    if isinstance(img_b, torch.Tensor) or (isinstance(img_b, np.ndarray) and img_b.max() <= 1.0):
        img_b = denormalize_image(img_b)
    
    # Handle batch dimension
    if attention_weights.ndim == 3:
        attention_weights = attention_weights[0]
    
    H, W = feature_size
    
    # Default grid positions (3x3 sampling)
    if grid_positions is None:
        grid_positions = [
            (H // 4, W // 4), (H // 4, W // 2), (H // 4, 3 * W // 4),
            (H // 2, W // 4), (H // 2, W // 2), (H // 2, 3 * W // 4),
            (3 * H // 4, W // 4), (3 * H // 4, W // 2), (3 * H // 4, 3 * W // 4),
        ]
    
    n_positions = len(grid_positions)
    n_cols = min(3, n_positions)
    n_rows = (n_positions + n_cols - 1) // n_cols
    
    # Create figure with extra column for source image
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(n_rows, n_cols + 1, width_ratios=[1] + [1] * n_cols)
    
    # Source image spanning all rows
    ax_source = fig.add_subplot(gs[:, 0])
    ax_source.imshow(img_a)
    ax_source.set_title('Source Image', fontsize=11)
    ax_source.axis('off')
    
    # Mark all source positions
    img_h, img_w = img_a.shape[:2]
    cell_h, cell_w = img_h / H, img_w / W
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_positions))
    for idx, (row, col) in enumerate(grid_positions):
        center_y = (row + 0.5) * cell_h
        center_x = (col + 0.5) * cell_w
        ax_source.plot(center_x, center_y, 'o', markersize=10, 
                      color=colors[idx], markeredgecolor='white', markeredgewidth=2)
        ax_source.text(center_x + 5, center_y, str(idx + 1), fontsize=8, 
                      color='white', fontweight='bold')
    
    # Reshape attention to spatial
    spatial_attn = reshape_attention_to_spatial(
        attention_weights[np.newaxis], feature_size
    )[0]
    
    # Plot attention for each position
    for idx, (row, col) in enumerate(grid_positions):
        ax_row = idx // n_cols
        ax_col = idx % n_cols + 1  # +1 for source image column
        
        ax = fig.add_subplot(gs[ax_row, ax_col])
        
        # Get attention map for this position
        attn_map = spatial_attn[row, col]  # (H, W)
        
        # Resize and normalize
        attn_resized = cv2.resize(attn_map, (img_b.shape[1], img_b.shape[0]))
        attn_resized = (attn_resized - attn_resized.min()) / (attn_resized.max() - attn_resized.min() + 1e-8)
        
        # Create overlay
        overlay = overlay_heatmap(img_b, attn_resized, alpha=0.6, colormap=ATTENTION_CMAP)
        
        ax.imshow(overlay)
        ax.set_title(f'Pos {idx + 1}: ({row}, {col})', fontsize=9, color=colors[idx])
        ax.axis('off')
    
    if title:
        fig.suptitle(title, fontsize=14, y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    return fig


def create_interpretation_figure(
    model: torch.nn.Module,
    img_a: torch.Tensor,
    img_b: torch.Tensor,
    label: int,
    device: torch.device = torch.device('cuda'),
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (16, 12),
) -> plt.Figure:
    """
    Create complete interpretability figure with GradCAM + attention.
    
    Args:
        model: VerificationModel instance
        img_a: (1, 3, H, W) or (3, H, W) first image tensor
        img_b: (1, 3, H, W) or (3, H, W) second image tensor
        label: Ground truth label (0 or 1)
        device: Device to run inference on
        save_path: Optional path to save figure
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    from .interpretability import VerificationGradCAM, AttentionExtractor
    
    # Ensure batch dimension and detach from any computation graph
    if img_a.dim() == 3:
        img_a = img_a.unsqueeze(0)
    if img_b.dim() == 3:
        img_b = img_b.unsqueeze(0)
    
    # Clone and detach to avoid grad issues
    img_a = img_a.clone().detach().to(device)
    img_b = img_b.clone().detach().to(device)
    model = model.to(device)
    model.eval()
    
    # Get prediction
    with torch.no_grad():
        logits = model(img_a, img_b)
        prediction = torch.sigmoid(logits).item()
    
    # Compute GradCAM
    gradcam = VerificationGradCAM(model)
    try:
        heatmap_a, heatmap_b = gradcam(img_a, img_b)
    finally:
        gradcam.remove_hooks()
    
    # Extract attention
    extractor = AttentionExtractor(model)
    try:
        attn_a2b, attn_b2a = extractor.get_cross_attention_maps(img_a, img_b, layer_idx=-1)
    finally:
        extractor.remove_hooks()
    
    # Denormalize images for plotting
    img_a_np = denormalize_image(img_a[0])
    img_b_np = denormalize_image(img_b[0])
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(3, 4, height_ratios=[1, 1, 1.2])
    
    # Row 1: Original images and GradCAM
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img_a_np)
    ax1.set_title('Image A', fontsize=11)
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(img_b_np)
    ax2.set_title('Image B', fontsize=11)
    ax2.axis('off')
    
    ax3 = fig.add_subplot(gs[0, 2])
    overlay_a = overlay_heatmap(img_a_np, heatmap_a[0])
    ax3.imshow(overlay_a)
    ax3.set_title('GradCAM A', fontsize=11)
    ax3.axis('off')
    
    ax4 = fig.add_subplot(gs[0, 3])
    overlay_b = overlay_heatmap(img_b_np, heatmap_b[0])
    ax4.imshow(overlay_b)
    ax4.set_title('GradCAM B', fontsize=11)
    ax4.axis('off')
    
    # Row 2: Cross-attention (aggregated)
    from .interpretability import aggregate_attention_map
    
    ax5 = fig.add_subplot(gs[1, 0:2])
    attn_agg_a2b = aggregate_attention_map(attn_a2b, (7, 7), 'mean')[0]
    attn_resized = cv2.resize(attn_agg_a2b, (img_b_np.shape[1], img_b_np.shape[0]))
    attn_resized = (attn_resized - attn_resized.min()) / (attn_resized.max() - attn_resized.min() + 1e-8)
    overlay_attn = overlay_heatmap(img_b_np, attn_resized, alpha=0.6, colormap=ATTENTION_CMAP)
    ax5.imshow(overlay_attn)
    ax5.set_title('Aggregated Attention A→B', fontsize=11)
    ax5.axis('off')
    
    ax6 = fig.add_subplot(gs[1, 2:4])
    attn_agg_b2a = aggregate_attention_map(attn_b2a, (7, 7), 'mean')[0]
    attn_resized = cv2.resize(attn_agg_b2a, (img_a_np.shape[1], img_a_np.shape[0]))
    attn_resized = (attn_resized - attn_resized.min()) / (attn_resized.max() - attn_resized.min() + 1e-8)
    overlay_attn = overlay_heatmap(img_a_np, attn_resized, alpha=0.6, colormap=ATTENTION_CMAP)
    ax6.imshow(overlay_attn)
    ax6.set_title('Aggregated Attention B→A', fontsize=11)
    ax6.axis('off')
    
    # Row 3: Attention from specific positions (center and corners)
    from .interpretability import reshape_attention_to_spatial
    
    positions = [(1, 1), (3, 3), (5, 5)]  # Top-left, center, bottom-right
    spatial_attn = reshape_attention_to_spatial(attn_a2b, (7, 7))[0]
    
    for i, (row, col) in enumerate(positions):
        ax = fig.add_subplot(gs[2, i])
        attn_map = spatial_attn[row, col]
        attn_resized = cv2.resize(attn_map, (img_b_np.shape[1], img_b_np.shape[0]))
        attn_resized = (attn_resized - attn_resized.min()) / (attn_resized.max() - attn_resized.min() + 1e-8)
        overlay = overlay_heatmap(img_b_np, attn_resized, alpha=0.6, colormap=ATTENTION_CMAP)
        ax.imshow(overlay)
        ax.set_title(f'A({row},{col})→B', fontsize=10)
        ax.axis('off')
    
    # Prediction info in last subplot
    ax_info = fig.add_subplot(gs[2, 3])
    ax_info.axis('off')
    
    pred_label = "Same" if prediction > 0.5 else "Different"
    true_label = "Same" if label == 1 else "Different"
    correct = (prediction > 0.5) == (label == 1)
    
    info_text = (
        f"Prediction: {pred_label}\n"
        f"Confidence: {prediction:.1%}\n"
        f"Ground Truth: {true_label}\n"
        f"Result: {'✓ Correct' if correct else '✗ Incorrect'}"
    )
    
    ax_info.text(
        0.5, 0.5, info_text,
        transform=ax_info.transAxes,
        fontsize=14,
        verticalalignment='center',
        horizontalalignment='center',
        bbox=dict(
            boxstyle='round',
            facecolor='lightgreen' if correct else 'lightcoral',
            alpha=0.8,
        ),
    )
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    return fig


def create_summary_grid(
    examples: List[Dict],
    n_cols: int = 4,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Optional[Tuple[int, int]] = None,
) -> plt.Figure:
    """
    Create summary grid of multiple examples for publication.
    
    Args:
        examples: List of dicts with keys:
            - 'img_a': (H, W, 3) uint8 image
            - 'img_b': (H, W, 3) uint8 image
            - 'heatmap_a': (H, W) heatmap
            - 'heatmap_b': (H, W) heatmap
            - 'prediction': float
            - 'label': int
            - 'category': str (e.g., 'TP', 'TN', 'FP', 'FN')
        n_cols: Number of columns
        save_path: Optional path to save figure
        figsize: Optional figure size
        
    Returns:
        matplotlib Figure
    """
    n_examples = len(examples)
    n_rows = (n_examples + n_cols - 1) // n_cols
    
    if figsize is None:
        figsize = (4 * n_cols, 3 * n_rows)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, example in enumerate(examples):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        # Create side-by-side with GradCAM overlay
        img_a = example['img_a']
        img_b = example['img_b']
        heatmap_a = example['heatmap_a']
        heatmap_b = example['heatmap_b']
        
        overlay_a = overlay_heatmap(img_a, heatmap_a, alpha=0.4)
        overlay_b = overlay_heatmap(img_b, heatmap_b, alpha=0.4)
        
        # Concatenate horizontally
        combined = np.concatenate([overlay_a, overlay_b], axis=1)
        ax.imshow(combined)
        
        # Title with prediction info
        pred = example['prediction']
        label = example['label']
        category = example.get('category', '')
        
        title_color = 'green' if (pred > 0.5) == (label == 1) else 'red'
        ax.set_title(f"{category}: {pred:.0%}", fontsize=10, color=title_color)
        ax.axis('off')
    
    # Hide empty subplots
    for idx in range(n_examples, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
    
    return fig

