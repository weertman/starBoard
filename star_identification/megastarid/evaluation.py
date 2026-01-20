"""
Detailed evaluation utilities for MegaStarID.

Provides per-identity and per-folder CMC evaluation with CSV output.
Supports TTA and re-ranking inference enhancements.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
from torch.amp import autocast
from tqdm import tqdm


def _get_autocast_context(device: torch.device):
    """Get appropriate autocast context for device (CPU-safe)."""
    if device.type == 'cuda':
        return autocast('cuda')
    return nullcontext()

from megastarid.inference import (
    extract_embeddings,
    extract_embeddings_with_tta,
    compute_similarity_matrix,
)


def _extract_with_metadata(
    model: nn.Module,
    dataloader,
    device: torch.device,
    use_tta: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Extract embeddings AND metadata in a single pass through the dataloader.
    
    This avoids the fragile pattern of iterating through the dataloader twice
    (once for metadata, once for embeddings) which could cause order mismatches
    if the dataloader ever shuffles.
    
    Args:
        model: The model to use for embedding extraction
        dataloader: DataLoader providing images, labels, identities, and paths
        device: Device to run inference on
        use_tta: Whether to use test-time augmentation (horizontal + vertical flip)
        
    Returns:
        embeddings: (N, D) numpy array of L2-normalized embeddings
        labels: (N,) numpy array of integer labels
        identities: List of N identity strings
        paths: List of N path strings
    """
    model.eval()
    
    all_embeddings = []
    all_labels = []
    all_identities = []
    all_paths = []
    
    num_augs = 3 if use_tta else 1  # original + h-flip + v-flip
    desc = f"Extracting embeddings (TTA x{num_augs})" if use_tta else "Extracting embeddings"
    
    with torch.no_grad(), _get_autocast_context(device):
        for batch in tqdm(dataloader, desc=desc, leave=False):
            images = batch['image'].to(device, non_blocking=True)
            
            # Original embedding
            emb = model(images, return_normalized=True)
            
            if use_tta:
                # Horizontal flip augmentation
                images_hflip = torch.flip(images, dims=[3])
                emb_hflip = model(images_hflip, return_normalized=True)
                
                # Vertical flip augmentation
                images_vflip = torch.flip(images, dims=[2])
                emb_vflip = model(images_vflip, return_normalized=True)
                
                # Average and re-normalize
                emb = (emb + emb_hflip + emb_vflip) / 3
                emb = F.normalize(emb, p=2, dim=1)
            
            all_embeddings.append(emb.cpu())
            all_labels.append(batch['label'])
            all_identities.extend(batch['identity'])
            all_paths.extend(batch['path'])
    
    embeddings = torch.cat(all_embeddings, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()
    
    return embeddings, labels, all_identities, all_paths


def _collect_metadata_and_labels(dataloader) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Collect labels and metadata from dataloader without running the model.
    
    Used to validate cached embeddings match current data order.
    
    Returns:
        labels: (N,) numpy array of integer labels
        identities: List of N identity strings
        paths: List of N path strings
    """
    all_labels = []
    all_identities = []
    all_paths = []
    
    for batch in dataloader:
        all_labels.append(batch['label'])
        all_identities.extend(batch['identity'])
        all_paths.extend(batch['path'])
    
    labels = torch.cat(all_labels, dim=0).numpy()
    return labels, all_identities, all_paths


def compute_detailed_star_metrics(
    model: nn.Module,
    gallery_loader,
    query_loader,
    device: torch.device,
    output_dir: Path,
    k_values: List[int] = [1, 5, 10, 20],
    use_tta: bool = False,
    use_reranking: bool = False,
    cached_embeddings: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Compute detailed per-identity and per-folder CMC metrics for star_dataset.
    
    Saves two CSV files:
    - star_identity_metrics.csv: Per-identity CMC performance
    - star_folder_metrics.csv: Per-folder (dataset) aggregated CMC performance
    
    Args:
        model: Trained model
        gallery_loader: DataLoader for gallery (train split)
        query_loader: DataLoader for query (test split)
        device: torch device
        output_dir: Directory to save CSV files
        k_values: K values for CMC@K computation
        use_tta: Whether to use test-time augmentation
        use_reranking: Whether to use k-reciprocal re-ranking
        cached_embeddings: Optional pre-computed embeddings dict to avoid 
                          re-extraction. Should contain 'gallery_embeddings',
                          'gallery_labels', 'query_embeddings', 'query_labels',
                          and optionally 'similarities'.
    
    Returns:
        Dictionary with summary metrics
    """
    model.eval()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use cached embeddings if provided (avoids re-extraction)
    if cached_embeddings is not None and not use_tta and not use_reranking:
        print("\nUsing cached embeddings...")
        gallery_embeddings = cached_embeddings['gallery_embeddings']
        cached_gallery_labels = cached_embeddings['gallery_labels']
        query_embeddings = cached_embeddings['query_embeddings']
        cached_query_labels = cached_embeddings['query_labels']
        
        # CRITICAL: Collect fresh metadata AND labels to validate cache matches current data
        # This single pass also gives us the identities and paths we need
        print("Validating cache against current data order...")
        fresh_gallery_labels, gallery_identities, gallery_paths = _collect_metadata_and_labels(gallery_loader)
        fresh_query_labels, query_identities, query_paths = _collect_metadata_and_labels(query_loader)
        
        # Validate that cached labels match fresh labels (catches stale cache bugs)
        if not np.array_equal(cached_gallery_labels, fresh_gallery_labels):
            raise ValueError(
                f"Cached gallery labels don't match current data order! "
                f"Cache has {len(cached_gallery_labels)} items, fresh data has {len(fresh_gallery_labels)}. "
                f"First mismatch at index {np.where(cached_gallery_labels != fresh_gallery_labels)[0][0] if len(cached_gallery_labels) == len(fresh_gallery_labels) else 'N/A (length mismatch)'}. "
                f"This can happen if the dataset config or seed changed. "
                f"Re-run without cached embeddings."
            )
        if not np.array_equal(cached_query_labels, fresh_query_labels):
            raise ValueError(
                f"Cached query labels don't match current data order! "
                f"Cache has {len(cached_query_labels)} items, fresh data has {len(fresh_query_labels)}. "
                f"Re-run without cached embeddings."
            )
        
        # Cache validated - use cached labels (they're identical to fresh)
        gallery_labels = cached_gallery_labels
        query_labels = cached_query_labels
        
        # Use cached similarities if available
        if 'similarities' in cached_embeddings:
            similarities = cached_embeddings['similarities']
        else:
            print("Computing similarities...")
            similarities = compute_similarity_matrix(
                query_embeddings=query_embeddings,
                gallery_embeddings=gallery_embeddings,
                use_reranking=False,
            )
    else:
        # Single-pass extraction: get embeddings AND metadata together
        # This eliminates the fragile double-iteration pattern
        print("\nExtracting embeddings...")
        gallery_embeddings, gallery_labels, gallery_identities, gallery_paths = _extract_with_metadata(
            model, gallery_loader, device, use_tta=use_tta
        )
        query_embeddings, query_labels, query_identities, query_paths = _extract_with_metadata(
            model, query_loader, device, use_tta=use_tta
        )
        
        # Compute similarity matrix (with or without re-ranking)
        print("Computing similarities...")
        similarities = compute_similarity_matrix(
            query_embeddings=query_embeddings,
            gallery_embeddings=gallery_embeddings,
            use_reranking=use_reranking,
        )
    
    # Per-query evaluation
    print("Computing per-identity metrics...")
    query_results = []
    skipped_queries = 0  # Track queries with no gallery matches
    
    for i in range(len(query_labels)):
        query_label = query_labels[i]
        query_identity = query_identities[i]
        query_path = query_paths[i]
        
        # Extract folder from path (parent directory name)
        # This is the outing/dataset folder that groups images
        folder = Path(query_path).parent.name
        
        # Get similarities for this query
        sims = similarities[i]
        
        # Sort gallery by similarity (descending)
        sorted_indices = np.argsort(-sims)
        sorted_labels = gallery_labels[sorted_indices]
        
        # Find matches (same identity)
        matches = sorted_labels == query_label
        
        if matches.sum() == 0:
            # No gallery samples for this identity
            skipped_queries += 1
            continue
        
        # Compute rank of first match
        first_match_idx = np.where(matches)[0][0]
        rank = first_match_idx + 1  # 1-indexed
        
        # Compute AP for this query
        cumsum = np.cumsum(matches)
        precision_at_k = cumsum / (np.arange(len(matches)) + 1)
        ap = (precision_at_k * matches).sum() / matches.sum()
        
        # CMC at various K
        cmc = {f'CMC@{k}': 1 if rank <= k else 0 for k in k_values}
        
        query_results.append({
            'identity': query_identity,
            'folder': folder,
            'query_path': query_path,
            'rank': rank,
            'AP': ap,
            'num_gallery_matches': int(matches.sum()),
            **cmc
        })
    
    # Log and validate evaluation results
    total_queries = len(query_labels)
    if skipped_queries > 0:
        print(f"  ⚠️ WARNING: {skipped_queries}/{total_queries} queries skipped (no gallery matches)")
    
    # Create DataFrame
    df_queries = pd.DataFrame(query_results)
    
    if len(df_queries) == 0:
        raise ValueError(
            f"Evaluation failed: ALL {total_queries} queries were skipped! "
            f"No query identity has matching samples in gallery. "
            f"Check your data split - this indicates a bug in train/test partitioning."
        )
    
    # === Per-Identity Metrics ===
    print("Aggregating per-identity metrics...")
    identity_metrics = df_queries.groupby('identity').agg({
        'folder': 'first',  # Folder for this identity
        'rank': ['mean', 'min', 'max', 'count'],
        'AP': 'mean',
        **{f'CMC@{k}': 'mean' for k in k_values}
    })
    
    # Flatten column names
    identity_metrics.columns = [
        'folder',
        'mean_rank', 'best_rank', 'worst_rank', 'num_queries',
        'mAP',
    ] + [f'CMC@{k}' for k in k_values]
    
    identity_metrics = identity_metrics.reset_index()
    identity_metrics = identity_metrics.sort_values('mAP', ascending=False)
    
    # Save per-identity CSV
    identity_csv_path = output_dir / 'star_identity_metrics.csv'
    identity_metrics.to_csv(identity_csv_path, index=False, float_format='%.4f')
    print(f"Saved per-identity metrics to: {identity_csv_path}")
    
    # === Per-Folder Metrics ===
    print("Aggregating per-folder metrics...")
    folder_metrics = df_queries.groupby('folder').agg({
        'identity': 'nunique',  # Number of unique identities
        'rank': ['mean', 'std', 'count'],
        'AP': 'mean',
        **{f'CMC@{k}': 'mean' for k in k_values}
    })
    
    # Flatten column names
    folder_metrics.columns = [
        'num_identities',
        'mean_rank', 'std_rank', 'num_queries',
        'mAP',
    ] + [f'CMC@{k}' for k in k_values]
    
    folder_metrics = folder_metrics.reset_index()
    folder_metrics = folder_metrics.sort_values('mAP', ascending=False)
    
    # Save per-folder CSV
    folder_csv_path = output_dir / 'star_folder_metrics.csv'
    folder_metrics.to_csv(folder_csv_path, index=False, float_format='%.4f')
    print(f"Saved per-folder metrics to: {folder_csv_path}")
    
    # === Overall Summary ===
    overall_metrics = {
        'mAP': float(df_queries['AP'].mean()),
        'mean_rank': float(df_queries['rank'].mean()),
        'num_queries': len(df_queries),
        'num_identities': df_queries['identity'].nunique(),
        'num_folders': df_queries['folder'].nunique(),
    }
    
    for k in k_values:
        overall_metrics[f'CMC@{k}'] = float(df_queries[f'CMC@{k}'].mean())
    
    # Print summary
    print("\n" + "=" * 60)
    print("STAR DATASET DETAILED EVALUATION")
    print("=" * 60)
    print(f"Queries: {overall_metrics['num_queries']}")
    print(f"Identities: {overall_metrics['num_identities']}")
    print(f"Folders: {overall_metrics['num_folders']}")
    print(f"\nOverall Performance:")
    print(f"  mAP: {overall_metrics['mAP']:.4f}")
    print(f"  Mean Rank: {overall_metrics['mean_rank']:.1f}")
    for k in k_values:
        print(f"  CMC@{k}: {overall_metrics[f'CMC@{k}']:.4f}")
    
    print(f"\nPer-Folder Summary:")
    for _, row in folder_metrics.iterrows():
        print(f"  {row['folder']}: mAP={row['mAP']:.4f}, CMC@1={row['CMC@1']:.4f} ({int(row['num_queries'])} queries)")
    
    # Save summary JSON
    import json
    summary_path = output_dir / 'star_evaluation_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(overall_metrics, f, indent=2)
    
    return overall_metrics


def evaluate_star_dataset(
    model: nn.Module,
    config,
    device: torch.device,
    output_dir: Optional[Path] = None,
    use_tta: Optional[bool] = None,
    use_reranking: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Convenience function to evaluate on star_dataset with detailed metrics.
    
    Args:
        model: Trained model
        config: Config with star_dataset_root and other settings
        device: torch device
        output_dir: Output directory (defaults to config.checkpoint_dir)
        use_tta: Override config.model.use_tta (if None, uses config)
        use_reranking: Override config.model.use_reranking (if None, uses config)
    
    Returns:
        Dictionary with evaluation metrics
    """
    from megastarid.datasets import create_finetune_dataloaders
    
    if output_dir is None:
        output_dir = Path(config.checkpoint_dir)
    
    # Get inference enhancement settings
    if use_tta is None:
        use_tta = getattr(config.model, 'use_tta', False)
    if use_reranking is None:
        use_reranking = getattr(config.model, 'use_reranking', False)
    
    # Create dataloaders
    _, gallery_loader, query_loader = create_finetune_dataloaders(config)
    
    # Run evaluation
    return compute_detailed_star_metrics(
        model=model,
        gallery_loader=gallery_loader,
        query_loader=query_loader,
        device=device,
        output_dir=output_dir,
        use_tta=use_tta,
        use_reranking=use_reranking,
    )

