#!/usr/bin/env python3
"""
TMP: Standalone t-SNE visualization demo for embedding space exploration.

This is a temporary/demo script - run directly to launch an interactive
Plotly visualization of gallery and query embeddings.

Usage:
    python -m src.dl.tmp_tsne_viz_demo
    
    Or from project root:
    python src/dl/tmp_tsne_viz_demo.py

Features:
- t-SNE dimensionality reduction of 512D embeddings to 2D
- Interactive scatter plot with hover info
- Gallery (circles) vs Query (triangles) distinction
- Click on query to see top-K similarity edges to gallery
- Thumbnail previews on hover (if available)
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
    from plotly.subplots import make_subplots
except ImportError:
    print("ERROR: plotly required. Run: pip install plotly")
    sys.exit(1)


def load_embeddings(model_key: str = None):
    """Load gallery and query embeddings from precomputed npz files."""
    from src.dl.registry import DLRegistry
    
    registry = DLRegistry.load()
    
    if model_key is None:
        model_key = registry.active_model
        if model_key is None:
            # Use first available
            if registry.models:
                model_key = next(iter(registry.models.keys()))
            else:
                raise ValueError("No models registered")
    
    model_dir = DLRegistry.get_model_data_dir(model_key)
    embeddings_dir = model_dir / "embeddings"
    
    gallery_path = embeddings_dir / "gallery_embeddings.npz"
    query_path = embeddings_dir / "query_embeddings.npz"
    
    if not gallery_path.exists() or not query_path.exists():
        raise FileNotFoundError(
            f"Embeddings not found. Run precomputation first.\n"
            f"Expected: {embeddings_dir}"
        )
    
    # Load embeddings
    gallery_data = np.load(gallery_path)
    query_data = np.load(query_path)
    
    gallery_ids = list(gallery_data.keys())
    query_ids = list(query_data.keys())
    
    gallery_embeddings = np.stack([gallery_data[gid] for gid in gallery_ids])
    query_embeddings = np.stack([query_data[qid] for qid in query_ids])
    
    print(f"Loaded {len(gallery_ids)} gallery, {len(query_ids)} query embeddings")
    print(f"Embedding dim: {gallery_embeddings.shape[1]}")
    
    return {
        "gallery_ids": gallery_ids,
        "query_ids": query_ids,
        "gallery_embeddings": gallery_embeddings,
        "query_embeddings": query_embeddings,
        "model_key": model_key
    }


def load_similarity_matrix(model_key: str):
    """Load the precomputed similarity matrix."""
    from src.dl.registry import DLRegistry
    
    model_dir = DLRegistry.get_model_data_dir(model_key)
    similarity_dir = model_dir / "similarity"
    
    matrix_path = similarity_dir / "query_gallery_scores.npz"
    mapping_path = similarity_dir / "id_mapping.json"
    
    if not matrix_path.exists():
        return None, None, None
    
    data = np.load(matrix_path)
    similarity = data['similarity']
    
    with open(mapping_path, 'r') as f:
        mapping = json.load(f)
    
    return similarity, mapping['query_ids'], mapping['gallery_ids']


def get_thumbnail_path(target: str, id_str: str) -> str:
    """Get path to first image for an ID (for thumbnail)."""
    try:
        from src.data.image_index import list_image_files
        images = list_image_files(target, id_str)
        if images:
            return str(images[0])
    except Exception:
        pass
    return ""


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
    print(f"Running t-SNE (perplexity={perplexity}, max_iter={config['max_iter']})...")
    
    n_samples = len(embeddings)
    # Adjust perplexity if too few samples
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


def create_visualization(data: dict, top_k: int = 5, tsne_config: dict = None):
    """Create interactive Plotly visualization.
    
    Args:
        data: Embedding data from load_embeddings()
        top_k: Number of top matches to show edges for
        tsne_config: Optional t-SNE configuration dict
    """
    
    gallery_ids = data["gallery_ids"]
    query_ids = data["query_ids"]
    gallery_emb = data["gallery_embeddings"]
    query_emb = data["query_embeddings"]
    model_key = data["model_key"]
    
    # Combine embeddings for t-SNE
    all_embeddings = np.vstack([gallery_emb, query_emb])
    all_ids = gallery_ids + query_ids
    all_types = ["Gallery"] * len(gallery_ids) + ["Query"] * len(query_ids)
    
    # Run t-SNE with config
    coords = run_tsne(all_embeddings, tsne_config=tsne_config)
    
    # Split coordinates
    gallery_coords = coords[:len(gallery_ids)]
    query_coords = coords[len(gallery_ids):]
    
    # Load similarity matrix for edges
    similarity, sim_query_ids, sim_gallery_ids = load_similarity_matrix(model_key)
    
    # Build hover text
    gallery_hover = [
        f"<b>{gid}</b><br>Type: Gallery<br>Index: {i}"
        for i, gid in enumerate(gallery_ids)
    ]
    query_hover = [
        f"<b>{qid}</b><br>Type: Query<br>Index: {i}"
        for i, qid in enumerate(query_ids)
    ]
    
    # Create figure
    fig = go.Figure()
    
    # Gallery points (circles)
    fig.add_trace(go.Scatter(
        x=gallery_coords[:, 0],
        y=gallery_coords[:, 1],
        mode='markers',
        marker=dict(
            size=12,
            color='#3498db',  # Blue
            symbol='circle',
            line=dict(width=1, color='white')
        ),
        text=gallery_hover,
        hoverinfo='text',
        name='Gallery',
        customdata=gallery_ids
    ))
    
    # Query points (triangles)
    fig.add_trace(go.Scatter(
        x=query_coords[:, 0],
        y=query_coords[:, 1],
        mode='markers',
        marker=dict(
            size=14,
            color='#e74c3c',  # Red
            symbol='triangle-up',
            line=dict(width=1, color='white')
        ),
        text=query_hover,
        hoverinfo='text',
        name='Query',
        customdata=query_ids
    ))
    
    # Add similarity edges (initially hidden, shown on click)
    # We'll add edges for each query to its top-K gallery matches
    if similarity is not None:
        # Create edge traces (one per query, initially invisible)
        edge_traces = []
        
        query_to_idx = {qid: i for i, qid in enumerate(sim_query_ids)}
        gallery_to_idx = {gid: i for i, gid in enumerate(sim_gallery_ids)}
        
        # Map gallery IDs to their t-SNE coordinates
        gallery_id_to_coord = {gid: gallery_coords[i] for i, gid in enumerate(gallery_ids)}
        
        for q_idx, qid in enumerate(query_ids):
            if qid not in query_to_idx:
                continue
            
            sim_q_idx = query_to_idx[qid]
            scores = similarity[sim_q_idx]
            
            # Get top-K gallery matches
            top_indices = np.argsort(scores)[::-1][:top_k]
            
            # Build edge lines
            edge_x = []
            edge_y = []
            edge_text = []
            
            query_coord = query_coords[q_idx]
            
            for g_idx in top_indices:
                gid = sim_gallery_ids[g_idx]
                if gid not in gallery_id_to_coord:
                    continue
                
                gallery_coord = gallery_id_to_coord[gid]
                score = scores[g_idx]
                
                # Line from query to gallery
                edge_x.extend([query_coord[0], gallery_coord[0], None])
                edge_y.extend([query_coord[1], gallery_coord[1], None])
                edge_text.append(f"{qid} → {gid}: {score:.3f}")
            
            if edge_x:
                edge_traces.append(go.Scatter(
                    x=edge_x,
                    y=edge_y,
                    mode='lines',
                    line=dict(width=2, color='rgba(255, 165, 0, 0.6)'),
                    hoverinfo='skip',
                    name=f'Edges: {qid}',
                    visible=False  # Hidden by default
                ))
        
        # Add all edge traces
        for trace in edge_traces:
            fig.add_trace(trace)
        
        # Add dropdown to select which query's edges to show
        if edge_traces:
            buttons = [dict(
                label="No edges",
                method="update",
                args=[{"visible": [True, True] + [False] * len(edge_traces)}]
            )]
            
            for i, qid in enumerate(query_ids[:len(edge_traces)]):
                visibility = [True, True] + [False] * len(edge_traces)
                visibility[2 + i] = True
                buttons.append(dict(
                    label=f"Query: {qid}",
                    method="update",
                    args=[{"visible": visibility}]
                ))
            
            fig.update_layout(
                updatemenus=[dict(
                    type="dropdown",
                    direction="down",
                    x=0.0,
                    y=1.15,
                    showactive=True,
                    buttons=buttons,
                    bgcolor='white',
                    font=dict(size=11)
                )]
            )
    
    # Layout
    fig.update_layout(
        title=dict(
            text=f"<b>Embedding Space Visualization</b><br><sup>Model: {model_key} | {len(gallery_ids)} Gallery, {len(query_ids)} Query</sup>",
            x=0.5,
            font=dict(size=18)
        ),
        xaxis=dict(
            title="t-SNE Dimension 1",
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.2)',
            zeroline=False
        ),
        yaxis=dict(
            title="t-SNE Dimension 2",
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.2)',
            zeroline=False
        ),
        legend=dict(
            x=1.02,
            y=1,
            bgcolor='rgba(255, 255, 255, 0.8)'
        ),
        hovermode='closest',
        plot_bgcolor='#fafafa',
        paper_bgcolor='white',
        width=1200,
        height=800,
        margin=dict(l=60, r=60, t=100, b=60)
    )
    
    return fig


def main():
    """Main entry point."""
    print("=" * 60)
    print("t-SNE Embedding Visualization Demo")
    print("=" * 60)
    print()
    
    try:
        # Load embeddings
        data = load_embeddings()
        
        # Create visualization
        fig = create_visualization(data, top_k=5)
        
        # Show in browser
        print("\nOpening visualization in browser...")
        fig.show()
        
        print("\nDone! Close the browser tab to exit.")
        
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("\nRun precomputation first:")
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

