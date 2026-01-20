#!/usr/bin/env python3
"""
TMP: Image-level t-SNE visualization demo.

Shows individual image embeddings (not identity centroids), revealing:
- Intra-identity variance (how spread out each identity's images are)
- Specific image-to-image relationships
- Outlier images within identities
- Best-matching image pairs across query/gallery

Usage:
    python -m src.dl.tmp_tsne_image_viz_demo
    
    Or from project root:
    python src/dl/tmp_tsne_image_viz_demo.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

# Check dependencies
try:
    from sklearn.manifold import TSNE
except ImportError:
    print("ERROR: scikit-learn required. Run: pip install scikit-learn")
    sys.exit(1)

try:
    import plotly.graph_objects as go
    import plotly.express as px
except ImportError:
    print("ERROR: plotly required. Run: pip install plotly")
    sys.exit(1)

try:
    from scipy import stats
except ImportError:
    print("ERROR: scipy required. Run: pip install scipy")
    sys.exit(1)


def load_image_embeddings(model_key: str = None):
    """Load per-image embeddings from precomputed npz files."""
    from src.dl.registry import DLRegistry
    
    registry = DLRegistry.load()
    
    if model_key is None:
        model_key = registry.active_model
        if model_key is None:
            if registry.models:
                model_key = next(iter(registry.models.keys()))
            else:
                raise ValueError("No models registered")
    
    model_dir = DLRegistry.get_model_data_dir(model_key)
    embeddings_dir = model_dir / "embeddings"
    
    # Load per-image embeddings
    gallery_img_path = embeddings_dir / "gallery_image_embeddings.npz"
    query_img_path = embeddings_dir / "query_image_embeddings.npz"
    gallery_paths_file = embeddings_dir / "gallery_image_paths.json"
    query_paths_file = embeddings_dir / "query_image_paths.json"
    
    if not gallery_img_path.exists() or not query_img_path.exists():
        raise FileNotFoundError(
            f"Per-image embeddings not found. Run precomputation first.\n"
            f"Expected: {embeddings_dir}"
        )
    
    # Load embeddings
    gallery_data = np.load(gallery_img_path)
    query_data = np.load(query_img_path)
    
    # Load paths
    with open(gallery_paths_file, 'r') as f:
        gallery_paths = json.load(f)
    with open(query_paths_file, 'r') as f:
        query_paths = json.load(f)
    
    # Build flat lists
    gallery_records = []  # [{id, path, embedding}, ...]
    query_records = []
    
    for gid in gallery_data.keys():
        embeddings = gallery_data[gid]
        paths = gallery_paths.get(gid, [])
        for i, emb in enumerate(embeddings):
            path = paths[i] if i < len(paths) else f"unknown_{i}"
            gallery_records.append({
                "id": gid,
                "path": path,
                "embedding": emb,
                "type": "Gallery"
            })
    
    for qid in query_data.keys():
        embeddings = query_data[qid]
        paths = query_paths.get(qid, [])
        for i, emb in enumerate(embeddings):
            path = paths[i] if i < len(paths) else f"unknown_{i}"
            query_records.append({
                "id": qid,
                "path": path,
                "embedding": emb,
                "type": "Query"
            })
    
    print(f"Loaded {len(gallery_records)} gallery images, {len(query_records)} query images")
    print(f"Embedding dim: {gallery_records[0]['embedding'].shape[0] if gallery_records else 'N/A'}")
    
    return {
        "gallery_records": gallery_records,
        "query_records": query_records,
        "model_key": model_key
    }


def run_tsne(embeddings: np.ndarray, tsne_config: dict = None):
    """Run t-SNE dimensionality reduction.
    
    Args:
        embeddings: Array of embeddings to reduce
        tsne_config: Optional dict with keys: perplexity, max_iter, learning_rate, init, random_state
    """
    # Default config
    config = {
        "perplexity": 30,
        "max_iter": 1000,
        "learning_rate": "auto",
        "init": "pca",
        "random_state": 42,
    }
    if tsne_config:
        config.update(tsne_config)
    
    perplexity = config["perplexity"]
    print(f"Running t-SNE on {len(embeddings)} images (perplexity={perplexity}, max_iter={config['max_iter']})...")
    
    n_samples = len(embeddings)
    effective_perplexity = min(perplexity, max(5, n_samples // 4))
    
    if effective_perplexity != perplexity:
        print(f"  Adjusted perplexity to {effective_perplexity} for {n_samples} samples")
    
    tsne = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        random_state=config["random_state"],
        max_iter=config["max_iter"],
        learning_rate=config["learning_rate"],
        init=config["init"]
    )
    
    coords = tsne.fit_transform(embeddings)
    print(f"  Done. Shape: {coords.shape}")
    return coords


def generate_colors(n: int):
    """Generate n distinct colors."""
    import colorsys
    colors = []
    for i in range(n):
        hue = i / n
        rgb = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
        hex_color = '#{:02x}{:02x}{:02x}'.format(
            int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
        )
        colors.append(hex_color)
    return colors


def compute_kde_contour(x_points: list, y_points: list, grid_size: int = 100, level: float = 0.95):
    """Compute 95% KDE contour for a set of 2D points.
    
    Args:
        x_points: List of x coordinates
        y_points: List of y coordinates
        grid_size: Resolution of the grid for contour computation
        level: Probability mass to contain (0.95 = 95% contour)
    
    Returns:
        Tuple of (X grid, Y grid, Z density, contour_level) or (None, None, None, None) if insufficient points
    """
    if len(x_points) < 3:  # Need minimum points for KDE
        return None, None, None, None
    
    x_arr = np.array(x_points)
    y_arr = np.array(y_points)
    
    # Create KDE
    try:
        xy = np.vstack([x_arr, y_arr])
        kde = stats.gaussian_kde(xy)
    except (np.linalg.LinAlgError, ValueError):
        # Singular matrix or other KDE failure (e.g., all points identical)
        return None, None, None, None
    
    # Create grid with padding around the points
    x_margin = (x_arr.max() - x_arr.min()) * 0.3 + 0.5
    y_margin = (y_arr.max() - y_arr.min()) * 0.3 + 0.5
    
    x_grid = np.linspace(x_arr.min() - x_margin, x_arr.max() + x_margin, grid_size)
    y_grid = np.linspace(y_arr.min() - y_margin, y_arr.max() + y_margin, grid_size)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # Evaluate KDE on grid
    positions = np.vstack([X.ravel(), Y.ravel()])
    Z = kde(positions).reshape(X.shape)
    
    # Find the contour level that contains 'level' probability mass
    # Sort density values and find threshold
    z_sorted = np.sort(Z.ravel())[::-1]
    z_cumsum = np.cumsum(z_sorted)
    z_cumsum = z_cumsum / z_cumsum[-1]  # Normalize to [0, 1]
    
    # Find the density threshold for the desired level
    idx = np.searchsorted(z_cumsum, level)
    if idx >= len(z_sorted):
        idx = len(z_sorted) - 1
    contour_level = z_sorted[idx]
    
    return X, Y, Z, contour_level


def create_visualization(data: dict, tsne_config: dict = None):
    """Create interactive Plotly visualization for image-level embeddings.
    
    Args:
        data: Embedding data from load_image_embeddings()
        tsne_config: Optional t-SNE configuration dict
    """
    
    gallery_records = data["gallery_records"]
    query_records = data["query_records"]
    model_key = data["model_key"]
    
    all_records = gallery_records + query_records
    
    if not all_records:
        print("No records to visualize!")
        return None
    
    # Combine embeddings
    all_embeddings = np.stack([r["embedding"] for r in all_records])
    
    # Run t-SNE with config
    coords = run_tsne(all_embeddings, tsne_config=tsne_config)
    
    # Assign coordinates back to records
    for i, record in enumerate(all_records):
        record["x"] = coords[i, 0]
        record["y"] = coords[i, 1]
    
    # Get unique identities for coloring
    all_ids = sorted(set(r["id"] for r in all_records))
    id_to_color = {id_str: color for id_str, color in zip(all_ids, generate_colors(len(all_ids)))}
    gallery_id_set = set(r["id"] for r in gallery_records)
    
    # Build figure - one trace per identity for highlight-on-hover
    fig = go.Figure()
    
    print("Building traces per identity for hover highlighting...")
    
    for id_str in all_ids:
        id_records = [r for r in all_records if r["id"] == id_str]
        if not id_records:
            continue
        
        is_gallery = id_str in gallery_id_set
        id_type = "Gallery" if is_gallery else "Query"
        color = id_to_color[id_str]
        
        # Image points for this identity
        x_vals = [r["x"] for r in id_records]
        y_vals = [r["y"] for r in id_records]
        hover_texts = [
            f"<b>{id_type}: {id_str}</b><br>"
            f"File: {Path(r['path']).name}"
            for r in id_records
        ]
        
        # Add image points trace
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_vals,
            mode='markers',
            marker=dict(
                size=10 if is_gallery else 8,
                color=color,
                symbol='circle' if is_gallery else 'triangle-up',
                line=dict(width=1, color='white' if is_gallery else 'black'),
                opacity=0.85
            ),
            text=hover_texts,
            hoverinfo='text',
            name=id_str,
            legendgroup=id_str,
            showlegend=True,
            visible=True
        ))
    
    # Track number of scatter traces (for dropdown visibility logic)
    n_scatter_traces = len(fig.data)
    
    # Add KDE contour traces for each identity (initially hidden)
    # Each identity gets multiple traces for glow effect: outer glow layers + main contour
    print("Computing KDE contours for each identity...")
    contour_info = []  # [(id_str, id_type, [trace_indices]), ...]
    
    # Glow layers: (width_multiplier, opacity)
    glow_layers = [
        (8.0, 0.15),   # Outermost glow - wide and faint
        (5.0, 0.25),   # Middle glow
        (3.0, 0.4),    # Inner glow
        (1.5, 1.0),    # Main contour line
    ]
    
    for id_str in all_ids:
        id_records = [r for r in all_records if r["id"] == id_str]
        if not id_records:
            continue
        
        is_gallery = id_str in gallery_id_set
        id_type = "Gallery" if is_gallery else "Query"
        color = id_to_color[id_str]
        
        x_vals = [r["x"] for r in id_records]
        y_vals = [r["y"] for r in id_records]
        
        # Compute KDE contour
        X, Y, Z, contour_level = compute_kde_contour(x_vals, y_vals)
        
        if X is not None:
            trace_indices = []
            
            # Add glow layers (outer to inner) + main contour
            for width_mult, opacity in glow_layers:
                # Convert hex color to rgba for opacity control
                hex_color = color.lstrip('#')
                r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                rgba_color = f'rgba({r}, {g}, {b}, {opacity})'
                
                fig.add_trace(go.Contour(
                    x=X[0, :],  # 1D array of x grid values
                    y=Y[:, 0],  # 1D array of y grid values
                    z=Z,
                    contours=dict(
                        start=contour_level,
                        end=contour_level,
                        size=1,  # Only one contour line
                        coloring='none'  # No fill, just lines
                    ),
                    line=dict(
                        color=rgba_color,
                        width=2.5 * width_mult,
                        dash='solid'
                    ),
                    showscale=False,
                    hoverinfo='skip',
                    name=f"95% KDE: {id_str}" if opacity == 1.0 else "",
                    visible=False  # Hidden by default
                ))
                trace_indices.append(len(fig.data) - 1)
            
            contour_info.append((id_str, id_type, trace_indices))
    
    # Build dropdown menu for contour selection
    if contour_info:
        # Count total contour traces (each identity has multiple glow layers)
        n_contour_traces = sum(len(indices) for _, _, indices in contour_info)
        
        # Base visibility: all scatter traces visible, all contours hidden
        base_visibility = [True] * n_scatter_traces + [False] * n_contour_traces
        
        buttons = [dict(
            label="No contours",
            method="update",
            args=[{"visible": base_visibility}]
        )]
        
        # Add button for each identity's contour (show all glow layers together)
        for id_str, id_type, trace_indices in contour_info:
            visibility = base_visibility.copy()
            # Show all traces for this identity (glow layers + main contour)
            for idx in trace_indices:
                visibility[idx] = True
            
            label = f"{id_type}: {id_str[:20]}..." if len(id_str) > 20 else f"{id_type}: {id_str}"
            buttons.append(dict(
                label=label,
                method="update",
                args=[{"visible": visibility}]
            ))
        
        fig.update_layout(
            updatemenus=[dict(
                type="dropdown",
                direction="down",
                x=0.0,
                y=1.12,
                xanchor="left",
                yanchor="top",
                showactive=True,
                buttons=buttons,
                bgcolor='white',
                bordercolor='gray',
                borderwidth=1,
                font=dict(size=11),
                pad=dict(l=10, r=10, t=5, b=5)
            )]
        )
        
        print(f"  Added {len(contour_info)} KDE contours (with glow effect) via dropdown selector")
    
    # Add stats annotations
    n_gallery_ids = len(gallery_id_set)
    n_query_ids = len(all_ids) - n_gallery_ids
    
    # Layout
    fig.update_layout(
        title=dict(
            text=(
                f"<b>Image-Level Embedding Space</b><br>"
                f"<sup>{len(gallery_records)} gallery images ({n_gallery_ids} IDs) | "
                f"{len(query_records)} query images ({n_query_ids} IDs) | Model: {model_key}</sup>"
            ),
            x=0.5,
            font=dict(size=16)
        ),
        xaxis=dict(
            title="t-SNE Dimension 1",
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.2)',
            zeroline=False,
            scaleanchor="y",
            scaleratio=1
        ),
        yaxis=dict(
            title="t-SNE Dimension 2",
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.2)',
            zeroline=False
        ),
        showlegend=False,
        hovermode='closest',
        hoverdistance=20,
        plot_bgcolor='#fafafa',
        paper_bgcolor='white',
        width=1400,
        height=900,
        margin=dict(l=60, r=200, t=100, b=60)
    )
    
    # Add annotation explaining visualization
    fig.add_annotation(
        text=(
            "<b>Symbols:</b><br>"
            "● Gallery images<br>"
            "▲ Query images<br>"
            "◉ 95% KDE contour (glow)<br><br>"
            "<b>Controls:</b><br>"
            "• Dropdown: show contour<br>"
            "• Hover for details<br>"
            "• Scroll to zoom<br>"
            "• Drag to pan<br><br>"
            "<b>Interpretation:</b><br>"
            "• Same color = same ID<br>"
            "• Tight cluster = consistent<br>"
            "• Query near gallery = match<br>"
            "• Contour = 95% density"
        ),
        xref="paper", yref="paper",
        x=1.01, y=0.5,
        showarrow=False,
        font=dict(size=11),
        align="left",
        bgcolor="rgba(255, 255, 255, 0.95)",
        bordercolor="gray",
        borderwidth=1,
        borderpad=10
    )
    
    return fig


def main():
    """Main entry point."""
    print("=" * 60)
    print("Image-Level t-SNE Visualization Demo")
    print("=" * 60)
    print()
    
    try:
        # Load per-image embeddings
        data = load_image_embeddings()
        
        # Create visualization
        fig = create_visualization(data)
        
        if fig is None:
            print("No visualization created.")
            sys.exit(1)
        
        # Show in browser (standard Plotly)
        print("\nOpening visualization in browser...")
        fig.show()
        
        print("\nDone! Hover over points to see details. Use mouse to zoom/pan.")
        
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("\nRun precomputation first to generate per-image embeddings:")
        print("  from src.dl.precompute import rerun_embeddings")
        print("  rerun_embeddings()")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

