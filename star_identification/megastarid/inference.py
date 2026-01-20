"""
Inference utilities for MegaStarID.

Includes:
- Test-Time Augmentation (TTA): horizontal flip averaging
- k-Reciprocal Re-ranking: post-hoc rank refinement
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext
from typing import Optional, Tuple, List, Union
from torch.utils.data import DataLoader
from torch.amp import autocast
from tqdm import tqdm


def _get_autocast_context(device: torch.device):
    """Get appropriate autocast context for device (CPU-safe)."""
    if device.type == 'cuda':
        return autocast('cuda')
    return nullcontext()


# =============================================================================
# Test-Time Augmentation
# =============================================================================

def extract_embeddings_with_tta(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    use_horizontal_flip: bool = True,
    use_vertical_flip: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract embeddings with test-time augmentation.
    
    Supports horizontal and vertical flip TTA:
    - Extract embedding from original image
    - Extract embedding from horizontally flipped image (if enabled)
    - Extract embedding from vertically flipped image (if enabled)
    - Average all embeddings
    - L2 normalize the result
    
    Args:
        model: The model to use for embedding extraction
        dataloader: DataLoader providing images and labels
        device: Device to run inference on
        use_horizontal_flip: Whether to use horizontal flip augmentation
        use_vertical_flip: Whether to use vertical flip augmentation
        
    Returns:
        embeddings: (N, D) array of embeddings
        labels: (N,) array of labels
    """
    model.eval()
    
    all_embeddings = []
    all_labels = []
    
    num_augs = 1 + int(use_horizontal_flip) + int(use_vertical_flip)
    desc = f"Extracting embeddings (TTA x{num_augs})"
    
    with torch.no_grad(), _get_autocast_context(device):
        for batch in tqdm(dataloader, desc=desc, leave=False):
            images = batch['image'].to(device, non_blocking=True)
            labels = batch['label']
            
            # Original embedding
            emb = model(images, return_normalized=True)
            
            if use_horizontal_flip:
                # Horizontal flip (flip width dimension)
                images_hflip = torch.flip(images, dims=[3])
                emb_hflip = model(images_hflip, return_normalized=True)
                emb = emb + emb_hflip
            
            if use_vertical_flip:
                # Vertical flip (flip height dimension)
                images_vflip = torch.flip(images, dims=[2])
                emb_vflip = model(images_vflip, return_normalized=True)
                emb = emb + emb_vflip
            
            # Average and re-normalize
            emb = emb / num_augs
            emb = F.normalize(emb, p=2, dim=1)
            
            all_embeddings.append(emb.cpu())
            all_labels.append(labels)
    
    embeddings = torch.cat(all_embeddings, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()
    
    return embeddings, labels


def extract_embeddings(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    keep_on_gpu: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract embeddings without TTA.
    
    Args:
        model: The model to use for embedding extraction
        dataloader: DataLoader providing images and labels
        device: Device to run inference on
        keep_on_gpu: If True, returns torch tensors on GPU instead of numpy
        
    Returns:
        embeddings: (N, D) array of embeddings (numpy or torch tensor)
        labels: (N,) array of labels (numpy or torch tensor)
    """
    model.eval()
    
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad(), _get_autocast_context(device):
        for batch in tqdm(dataloader, desc="Extracting embeddings", leave=False):
            images = batch['image'].to(device, non_blocking=True)
            labels = batch['label']
            
            emb = model(images, return_normalized=True)
            
            if keep_on_gpu:
                all_embeddings.append(emb)
            else:
                all_embeddings.append(emb.cpu())
            all_labels.append(labels)
    
    embeddings = torch.cat(all_embeddings, dim=0)
    labels = torch.cat(all_labels, dim=0)
    
    if not keep_on_gpu:
        embeddings = embeddings.numpy()
        labels = labels.numpy()
    
    return embeddings, labels


# =============================================================================
# k-Reciprocal Re-ranking
# =============================================================================

def _build_k_reciprocal_encoding(
    all_rank: np.ndarray,
    original_dist: np.ndarray,
    k1: int,
) -> np.ndarray:
    """
    Build k-reciprocal encoding matrix V (vectorized implementation).
    
    Uses precomputed top-k membership matrices to avoid nested loops
    for reciprocal neighbor checking.
    
    Args:
        all_rank: (num_all, num_all) ranking indices for each sample
        original_dist: (num_all, num_all) pairwise distance matrix
        k1: k for k-reciprocal neighbors
        
    Returns:
        V: (num_all, num_all) k-reciprocal encoding matrix
    """
    num_all = all_rank.shape[0]
    V = np.zeros((num_all, num_all), dtype=np.float32)
    
    # Precompute top-k membership matrices (vectorized)
    # is_in_topk[i, j] = True if j is in i's top-k1 neighbors
    topk_neighbors = all_rank[:, :k1 + 1]  # (num_all, k1+1)
    is_in_topk = np.zeros((num_all, num_all), dtype=bool)
    rows = np.repeat(np.arange(num_all), k1 + 1)
    cols = topk_neighbors.ravel()
    is_in_topk[rows, cols] = True
    
    # Similarly for k1/2 neighbors (used in expansion)
    k1_half = int(k1 / 2) + 1
    topk_half_neighbors = all_rank[:, :k1_half]
    is_in_topk_half = np.zeros((num_all, num_all), dtype=bool)
    rows_half = np.repeat(np.arange(num_all), k1_half)
    cols_half = topk_half_neighbors.ravel()
    is_in_topk_half[rows_half, cols_half] = True
    
    # k-reciprocal: mutual top-k neighbors (vectorized)
    is_k_reciprocal = is_in_topk & is_in_topk.T
    
    # Build V matrix with expansion (still needs loop for expansion logic)
    for i in range(num_all):
        # Get k-reciprocal neighbors for i
        k_reciprocal_indices = np.where(is_k_reciprocal[i])[0]
        
        if len(k_reciprocal_indices) == 0:
            continue
        
        # Expansion: for each k-reciprocal neighbor, check their half-k neighbors
        expansion_set = set(k_reciprocal_indices)
        
        for candidate in k_reciprocal_indices:
            # Get candidate's k1/2 neighbors that are also reciprocal with candidate
            candidate_half_neighbors = np.where(is_in_topk_half[candidate])[0]
            # Check which are reciprocal with candidate
            reciprocal_mask = is_in_topk_half[candidate_half_neighbors, candidate]
            candidate_reciprocal = candidate_half_neighbors[reciprocal_mask]
            
            # Expansion criterion: > 2/3 overlap
            if len(candidate_reciprocal) > 2 / 3 * len(candidate_half_neighbors):
                expansion_set.update(candidate_reciprocal)
        
        k_reciprocal_expansion_index = np.array(list(expansion_set))
        
        # Gaussian weighted encoding
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = weight / (np.sum(weight) + 1e-8)
    
    return V


def _compute_jaccard_vectorized(
    V: np.ndarray,
    num_query: int,
    num_gallery: int,
    chunk_size: int = 100,
) -> np.ndarray:
    """
    Compute Jaccard distance using vectorized operations with chunking.
    
    Processes in chunks to avoid memory blowup on large datasets.
    
    Args:
        V: (num_all, num_all) k-reciprocal encoding matrix
        num_query: Number of query samples
        num_gallery: Number of gallery samples
        chunk_size: Chunk size for memory-efficient processing
        
    Returns:
        jaccard_dist: (num_query, num_gallery) Jaccard distance matrix
    """
    V_query = V[:num_query]
    V_gallery = V[num_query:num_query + num_gallery]
    
    jaccard_dist = np.zeros((num_query, num_gallery), dtype=np.float32)
    
    # Process in chunks to avoid memory blowup
    for i in range(0, num_query, chunk_size):
        i_end = min(i + chunk_size, num_query)
        V_q_chunk = V_query[i:i_end]  # (chunk, num_all)
        
        # Vectorized min/max over all gallery for this query chunk
        # Shape: (chunk, num_gallery, num_all)
        minimum = np.minimum(V_q_chunk[:, None, :], V_gallery[None, :, :])
        maximum = np.maximum(V_q_chunk[:, None, :], V_gallery[None, :, :])
        
        min_sum = minimum.sum(axis=2)  # (chunk, num_gallery)
        max_sum = maximum.sum(axis=2)  # (chunk, num_gallery)
        
        jaccard_dist[i:i_end] = 1 - min_sum / (max_sum + 1e-8)
    
    return jaccard_dist


def compute_jaccard_distance(
    initial_rank: np.ndarray,
    query_features: np.ndarray,
    gallery_features: np.ndarray,
    k1: int = 20,
    k2: int = 6,
    lambda_value: float = 0.3,
) -> np.ndarray:
    """
    Compute Jaccard distance with k-reciprocal re-ranking.
    
    Reference:
        Zhong et al. "Re-ranking Person Re-identification with k-reciprocal Encoding"
        CVPR 2017
    
    Args:
        initial_rank: (num_query, num_gallery) initial ranking indices
        query_features: (num_query, D) query embeddings
        gallery_features: (num_gallery, D) gallery embeddings
        k1: k for k-reciprocal neighbors
        k2: k for expanded k-reciprocal neighbors
        lambda_value: weight for original distance
        
    Returns:
        final_dist: (num_query, num_gallery) refined distance matrix
    """
    num_query = query_features.shape[0]
    num_gallery = gallery_features.shape[0]
    num_all = num_query + num_gallery
    
    # Concatenate query and gallery features
    all_features = np.concatenate([query_features, gallery_features], axis=0)
    
    # Compute original distance (cosine distance = 1 - cosine similarity)
    original_dist = 1 - np.dot(all_features, all_features.T)
    original_dist = np.clip(original_dist, 0, 2)  # Handle numerical issues
    
    # Get initial ranking for all samples
    all_rank = np.argsort(original_dist, axis=1)
    
    # Build V matrix using vectorized k-reciprocal encoding
    V = _build_k_reciprocal_encoding(all_rank, original_dist, k1)
    
    # Local query expansion (vectorized)
    if k2 > 0:
        # Gather neighbors for each sample and average their V rows
        neighbor_indices = all_rank[:, :k2 + 1]  # (num_all, k2+1)
        V_qe = V[neighbor_indices].mean(axis=1)  # (num_all, num_all)
        V = V_qe
    
    # Compute Jaccard distance (vectorized with chunking)
    jaccard_dist = _compute_jaccard_vectorized(V, num_query, num_gallery)
    
    # Final distance: combine Jaccard and original
    final_dist = jaccard_dist * (1 - lambda_value) + original_dist[:num_query, num_query:] * lambda_value
    
    return final_dist


def rerank(
    query_embeddings: np.ndarray,
    gallery_embeddings: np.ndarray,
    k1: int = 20,
    k2: int = 6,
    lambda_value: float = 0.3,
) -> np.ndarray:
    """
    Apply k-reciprocal re-ranking to refine similarity scores.
    
    Args:
        query_embeddings: (num_query, D) L2-normalized query embeddings
        gallery_embeddings: (num_gallery, D) L2-normalized gallery embeddings
        k1: k for k-reciprocal neighbors (default: 20)
        k2: k for local query expansion (default: 6)
        lambda_value: weight for original distance (default: 0.3)
        
    Returns:
        refined_dist: (num_query, num_gallery) refined distance matrix
                     (lower = more similar)
    """
    # Initial ranking based on cosine similarity
    initial_sim = np.dot(query_embeddings, gallery_embeddings.T)
    initial_rank = np.argsort(-initial_sim, axis=1)  # Descending similarity
    
    # Compute refined distance with re-ranking
    refined_dist = compute_jaccard_distance(
        initial_rank=initial_rank,
        query_features=query_embeddings,
        gallery_features=gallery_embeddings,
        k1=k1,
        k2=k2,
        lambda_value=lambda_value,
    )
    
    return refined_dist


# =============================================================================
# Combined Inference Pipeline
# =============================================================================

def compute_similarity_matrix(
    query_embeddings: np.ndarray,
    gallery_embeddings: np.ndarray,
    use_reranking: bool = False,
    k1: int = 20,
    k2: int = 6,
    lambda_value: float = 0.3,
) -> np.ndarray:
    """
    Compute similarity matrix between query and gallery.
    
    Args:
        query_embeddings: (num_query, D) embeddings
        gallery_embeddings: (num_gallery, D) embeddings
        use_reranking: Whether to apply k-reciprocal re-ranking
        k1, k2, lambda_value: Re-ranking hyperparameters
        
    Returns:
        similarity: (num_query, num_gallery) similarity matrix
                   (higher = more similar)
    """
    if use_reranking:
        # Re-ranking returns distance (lower = better)
        distance = rerank(
            query_embeddings=query_embeddings,
            gallery_embeddings=gallery_embeddings,
            k1=k1,
            k2=k2,
            lambda_value=lambda_value,
        )
        # Convert distance to similarity
        similarity = 1 - distance
    else:
        # Simple cosine similarity
        similarity = np.dot(query_embeddings, gallery_embeddings.T)
    
    return similarity


def compute_reid_metrics_with_enhancements(
    model: nn.Module,
    gallery_loader: DataLoader,
    query_loader: DataLoader,
    device: torch.device,
    use_tta: bool = False,
    use_reranking: bool = False,
    k1: int = 20,
    k2: int = 6,
    lambda_value: float = 0.3,
) -> dict:
    """
    Compute re-identification metrics with optional enhancements.
    
    Args:
        model: The model to evaluate
        gallery_loader: DataLoader for gallery images
        query_loader: DataLoader for query images
        device: Device to run on
        use_tta: Whether to use test-time augmentation
        use_reranking: Whether to use k-reciprocal re-ranking
        k1, k2, lambda_value: Re-ranking hyperparameters
        
    Returns:
        Dict with mAP, Rank-1, Rank-5, Rank-10 metrics
    """
    import torch
    
    # For non-reranking path, keep embeddings on GPU for faster similarity
    keep_on_gpu = not use_reranking and device.type == 'cuda'
    
    # Extract embeddings
    if use_tta:
        gallery_embeddings, gallery_labels = extract_embeddings_with_tta(
            model, gallery_loader, device,
            use_horizontal_flip=True, use_vertical_flip=True
        )
        query_embeddings, query_labels = extract_embeddings_with_tta(
            model, query_loader, device,
            use_horizontal_flip=True, use_vertical_flip=True
        )
        # TTA path always returns numpy
        keep_on_gpu = False
    else:
        gallery_embeddings, gallery_labels = extract_embeddings(
            model, gallery_loader, device, keep_on_gpu=keep_on_gpu
        )
        query_embeddings, query_labels = extract_embeddings(
            model, query_loader, device, keep_on_gpu=keep_on_gpu
        )
    
    # Compute similarity - use GPU if available
    if keep_on_gpu and isinstance(gallery_embeddings, torch.Tensor):
        # GPU path: compute similarity on GPU, then move to CPU
        with torch.no_grad():
            similarity = (query_embeddings @ gallery_embeddings.T).cpu().numpy()
        # Convert labels to numpy
        gallery_labels = gallery_labels.numpy()
        query_labels = query_labels.numpy()
    else:
        # CPU path or reranking path
        if isinstance(gallery_embeddings, torch.Tensor):
            gallery_embeddings = gallery_embeddings.numpy()
            query_embeddings = query_embeddings.numpy()
            gallery_labels = gallery_labels.numpy()
            query_labels = query_labels.numpy()
        
        similarity = compute_similarity_matrix(
            query_embeddings=query_embeddings,
            gallery_embeddings=gallery_embeddings,
            use_reranking=use_reranking,
            k1=k1,
            k2=k2,
            lambda_value=lambda_value,
        )
    
    # Compute metrics
    all_aps = []
    all_ranks = []
    skipped_queries = 0  # Track queries with no gallery matches
    
    for i in range(len(query_labels)):
        query_label = query_labels[i]
        sims = similarity[i]
        
        # Sort by similarity (descending)
        sorted_indices = np.argsort(-sims)
        sorted_labels = gallery_labels[sorted_indices]
        
        matches = sorted_labels == query_label
        
        if matches.sum() == 0:
            skipped_queries += 1
            continue
        
        # Average Precision
        cumsum = np.cumsum(matches)
        precision = cumsum / (np.arange(len(matches)) + 1)
        ap = (precision * matches).sum() / matches.sum()
        all_aps.append(ap)
        
        # Rank
        first_match = np.where(matches)[0][0]
        all_ranks.append(first_match + 1)
    
    # Log and validate evaluation results
    total_queries = len(query_labels)
    if skipped_queries > 0:
        print(f"  ⚠️ WARNING: {skipped_queries}/{total_queries} queries skipped (no gallery matches)")
    
    if len(all_aps) == 0:
        raise ValueError(
            f"Evaluation failed: ALL {total_queries} queries were skipped! "
            f"No query identity has matching samples in gallery. "
            f"Check your data split - this indicates a bug in train/test partitioning."
        )
    
    metrics = {
        'mAP': float(np.mean(all_aps)),
        'Rank-1': float(np.mean([1.0 if r <= 1 else 0.0 for r in all_ranks])),
        'Rank-5': float(np.mean([1.0 if r <= 5 else 0.0 for r in all_ranks])),
        'Rank-10': float(np.mean([1.0 if r <= 10 else 0.0 for r in all_ranks])),
        'num_valid_queries': len(all_aps),
        'num_skipped_queries': skipped_queries,
    }
    
    return metrics

