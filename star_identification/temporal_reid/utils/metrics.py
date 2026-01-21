"""
Evaluation metrics for re-identification.

Key metrics:
- Rank-k accuracy: Is the correct identity in top k results?
- mAP: Mean Average Precision
- CMC: Cumulative Matching Characteristic
"""
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm


@torch.no_grad()
def extract_embeddings(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Extract embeddings from a dataloader.
    
    Args:
        model: Trained model
        dataloader: DataLoader to extract from
        device: torch device
    
    Returns:
        embeddings: (N, D) numpy array
        labels: (N,) numpy array
        identities: List of identity strings
    """
    model.eval()
    
    all_embeddings = []
    all_labels = []
    all_identities = []
    
    for batch in tqdm(dataloader, desc="Extracting embeddings", leave=False):
        images = batch['image'].to(device)
        labels = batch['label']
        identities = batch['identity']
        
        embeddings = model(images, return_normalized=True)
        
        all_embeddings.append(embeddings.cpu().numpy())
        all_labels.append(labels.numpy())
        all_identities.extend(identities)
    
    embeddings = np.vstack(all_embeddings)
    labels = np.concatenate(all_labels)
    
    return embeddings, labels, all_identities


def compute_distance_matrix(
    query_embeddings: np.ndarray,
    gallery_embeddings: np.ndarray,
    metric: str = 'cosine',
) -> np.ndarray:
    """
    Compute pairwise distance matrix.
    
    Args:
        query_embeddings: (Q, D)
        gallery_embeddings: (G, D)
        metric: 'cosine' or 'euclidean'
    
    Returns:
        distances: (Q, G) distance matrix (lower = more similar)
    """
    if metric == 'cosine':
        # Cosine distance = 1 - cosine similarity
        query_norm = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)
        gallery_norm = gallery_embeddings / np.linalg.norm(gallery_embeddings, axis=1, keepdims=True)
        similarity = query_norm @ gallery_norm.T
        distances = 1 - similarity
    elif metric == 'euclidean':
        # Squared Euclidean distance
        q_sq = np.sum(query_embeddings ** 2, axis=1, keepdims=True)
        g_sq = np.sum(gallery_embeddings ** 2, axis=1, keepdims=True)
        distances = q_sq + g_sq.T - 2 * query_embeddings @ gallery_embeddings.T
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    return distances


def compute_cmc(
    distances: np.ndarray,
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
    max_rank: int = 10,
) -> np.ndarray:
    """
    Compute Cumulative Matching Characteristic (CMC) curve.
    
    Args:
        distances: (Q, G) distance matrix
        query_labels: (Q,) query identity labels
        gallery_labels: (G,) gallery identity labels
        max_rank: Maximum rank to compute
    
    Returns:
        cmc: (max_rank,) CMC curve values
    """
    num_queries = distances.shape[0]
    cmc = np.zeros(max_rank)
    
    # Sort gallery by distance for each query
    indices = np.argsort(distances, axis=1)
    
    for q_idx in range(num_queries):
        q_label = query_labels[q_idx]
        
        # Get sorted gallery labels for this query
        sorted_labels = gallery_labels[indices[q_idx]]
        
        # Find first correct match
        matches = (sorted_labels == q_label)
        first_match = np.where(matches)[0]
        
        if len(first_match) > 0:
            first_match_rank = first_match[0]
            if first_match_rank < max_rank:
                cmc[first_match_rank:] += 1
    
    cmc = cmc / num_queries
    return cmc


def compute_map(
    distances: np.ndarray,
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
) -> float:
    """
    Compute Mean Average Precision (mAP).
    
    Args:
        distances: (Q, G) distance matrix
        query_labels: (Q,) query identity labels
        gallery_labels: (G,) gallery identity labels
    
    Returns:
        mAP: Mean Average Precision
    """
    num_queries = distances.shape[0]
    all_ap = []
    
    # Sort gallery by distance for each query
    indices = np.argsort(distances, axis=1)
    
    for q_idx in range(num_queries):
        q_label = query_labels[q_idx]
        
        # Get sorted gallery labels
        sorted_labels = gallery_labels[indices[q_idx]]
        
        # Find matches
        matches = (sorted_labels == q_label).astype(float)
        
        if matches.sum() == 0:
            # No matches in gallery for this query
            continue
        
        # Compute AP
        cumsum = np.cumsum(matches)
        precision_at_k = cumsum / (np.arange(len(matches)) + 1)
        ap = (precision_at_k * matches).sum() / matches.sum()
        all_ap.append(ap)
    
    if len(all_ap) == 0:
        return 0.0
    
    return np.mean(all_ap)


def compute_reid_metrics(
    model: torch.nn.Module,
    query_loader,
    gallery_loader,
    device: torch.device,
    max_rank: int = 10,
) -> Dict[str, float]:
    """
    Compute full re-identification metrics.
    
    This is the main evaluation function. It:
    1. Extracts embeddings from query and gallery sets
    2. Computes distance matrix
    3. Computes CMC and mAP metrics
    
    Args:
        model: Trained model
        query_loader: DataLoader for query images (test outings)
        gallery_loader: DataLoader for gallery images (train outings)
        device: torch device
        max_rank: Maximum rank for CMC
    
    Returns:
        Dictionary of metrics:
        - rank_1, rank_5, rank_10: CMC accuracies
        - mAP: Mean Average Precision
    """
    # Extract embeddings
    query_emb, query_labels, query_ids = extract_embeddings(model, query_loader, device)
    gallery_emb, gallery_labels, gallery_ids = extract_embeddings(model, gallery_loader, device)
    
    print(f"\nEvaluation:")
    print(f"  Query: {len(query_emb)} images, {len(set(query_labels))} identities")
    print(f"  Gallery: {len(gallery_emb)} images, {len(set(gallery_labels))} identities")
    
    # Compute distances
    distances = compute_distance_matrix(query_emb, gallery_emb, metric='cosine')
    
    # Compute CMC
    cmc = compute_cmc(distances, query_labels, gallery_labels, max_rank=max_rank)
    
    # Compute mAP
    mAP = compute_map(distances, query_labels, gallery_labels)
    
    metrics = {
        'rank_1': cmc[0],
        'rank_5': cmc[4] if max_rank >= 5 else cmc[-1],
        'rank_10': cmc[9] if max_rank >= 10 else cmc[-1],
        'mAP': mAP,
    }
    
    print(f"  Rank-1: {metrics['rank_1']:.4f}")
    print(f"  Rank-5: {metrics['rank_5']:.4f}")
    print(f"  Rank-10: {metrics['rank_10']:.4f}")
    print(f"  mAP: {metrics['mAP']:.4f}")
    
    return metrics



