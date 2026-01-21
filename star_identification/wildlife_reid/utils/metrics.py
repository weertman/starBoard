"""
Evaluation metrics for Wildlife ReID.
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict
from tqdm import tqdm


def compute_reid_metrics(
    model,
    test_loader,
    device: torch.device,
    k_values: list = [1, 5, 10],
) -> Dict[str, float]:
    """
    Compute ReID metrics: CMC (Rank-k) and mAP.
    
    For Wildlife10k, we use the test set as both query and gallery.
    Each sample queries against all other samples.
    
    Args:
        model: Trained model
        test_loader: Test dataloader
        device: Device to use
        k_values: k values for Rank-k accuracy
    
    Returns:
        Dictionary with mAP and Rank-k metrics
    """
    model.eval()
    
    all_embeddings = []
    all_labels = []
    all_datasets = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Extracting embeddings", leave=False):
            images = batch['image'].to(device)
            labels = batch['label']
            datasets = batch.get('dataset', ['unknown'] * len(labels))
            
            embeddings = model(images, return_normalized=True)
            
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels)
            all_datasets.extend(datasets)
    
    embeddings = torch.cat(all_embeddings, dim=0)  # (N, D)
    labels = torch.cat(all_labels, dim=0)  # (N,)
    
    # Compute pairwise similarities
    similarities = embeddings @ embeddings.t()  # (N, N)
    
    # Convert to numpy for easier manipulation
    similarities = similarities.numpy()
    labels = labels.numpy()
    
    n_samples = len(labels)
    
    # For each query, rank all gallery samples
    all_aps = []
    all_ranks = []
    
    for i in range(n_samples):
        query_label = labels[i]
        
        # Get similarities (exclude self)
        sims = similarities[i].copy()
        sims[i] = -np.inf  # Exclude self
        
        # Sort by similarity (descending)
        sorted_indices = np.argsort(-sims)
        sorted_labels = labels[sorted_indices]
        
        # Find matches (same identity)
        matches = (sorted_labels == query_label)
        
        if matches.sum() == 0:
            # No positive samples (this identity only has 1 image)
            continue
        
        # Compute AP
        ap = compute_ap(matches)
        all_aps.append(ap)
        
        # Compute rank of first match
        first_match = np.where(matches)[0][0]
        all_ranks.append(first_match + 1)  # 1-indexed
    
    # Aggregate metrics
    metrics = {
        'mAP': np.mean(all_aps) if all_aps else 0.0,
    }
    
    # CMC (Cumulative Matching Characteristics)
    for k in k_values:
        rank_k_acc = np.mean([1.0 if r <= k else 0.0 for r in all_ranks]) if all_ranks else 0.0
        metrics[f'Rank-{k}'] = rank_k_acc
    
    return metrics


def compute_ap(matches: np.ndarray) -> float:
    """
    Compute Average Precision for a single query.
    
    Args:
        matches: Boolean array where True indicates a match
    
    Returns:
        AP score
    """
    n_pos = matches.sum()
    if n_pos == 0:
        return 0.0
    
    cumsum = np.cumsum(matches)
    precision = cumsum / (np.arange(len(matches)) + 1)
    
    # Only consider precision at recall points (where matches occur)
    ap = (precision * matches).sum() / n_pos
    
    return ap


def compute_per_dataset_metrics(
    model,
    test_loader,
    device: torch.device,
    k_values: list = [1, 5, 10],
) -> Dict[str, Dict[str, float]]:
    """
    Compute per-dataset metrics.
    
    Returns dictionary with overall metrics and per-dataset metrics.
    """
    model.eval()
    
    all_embeddings = []
    all_labels = []
    all_datasets = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Extracting embeddings", leave=False):
            images = batch['image'].to(device)
            labels = batch['label']
            datasets = batch['dataset']
            
            embeddings = model(images, return_normalized=True)
            
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels)
            all_datasets.extend(datasets)
    
    embeddings = torch.cat(all_embeddings, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()
    datasets = np.array(all_datasets)
    
    results = {}
    
    # Overall metrics
    results['overall'] = _compute_metrics_for_subset(
        embeddings, labels, np.ones(len(labels), dtype=bool), k_values
    )
    
    # Per-dataset metrics
    unique_datasets = np.unique(datasets)
    for ds in unique_datasets:
        mask = datasets == ds
        if mask.sum() >= 2:  # Need at least 2 samples
            results[ds] = _compute_metrics_for_subset(
                embeddings, labels, mask, k_values
            )
    
    return results


def _compute_metrics_for_subset(
    embeddings: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
    k_values: list,
) -> Dict[str, float]:
    """Compute metrics for a subset of samples."""
    subset_embeddings = embeddings[mask]
    subset_labels = labels[mask]
    
    # Compute similarities
    similarities = subset_embeddings @ subset_embeddings.T
    
    n_samples = len(subset_labels)
    all_aps = []
    all_ranks = []
    
    for i in range(n_samples):
        query_label = subset_labels[i]
        
        sims = similarities[i].copy()
        sims[i] = -np.inf
        
        sorted_indices = np.argsort(-sims)
        sorted_labels = subset_labels[sorted_indices]
        
        matches = (sorted_labels == query_label)
        
        if matches.sum() == 0:
            continue
        
        ap = compute_ap(matches)
        all_aps.append(ap)
        
        first_match = np.where(matches)[0][0]
        all_ranks.append(first_match + 1)
    
    metrics = {
        'mAP': np.mean(all_aps) if all_aps else 0.0,
        'n_queries': len(all_aps),
    }
    
    for k in k_values:
        rank_k_acc = np.mean([1.0 if r <= k else 0.0 for r in all_ranks]) if all_ranks else 0.0
        metrics[f'Rank-{k}'] = rank_k_acc
    
    return metrics


