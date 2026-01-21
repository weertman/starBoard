"""
Outlier detection for identity embeddings.

Uses nearest-neighbor similarity with robust statistics (median/MAD) to detect
embeddings that don't belong to an identity. This approach:
- Handles multi-modal distributions (e.g., front vs back views of same animal)
- Is robust to outlier contamination (uses median/MAD, not mean/std)
- Works across different model types (triplet loss, circle loss, etc.)

The key insight: an outlier is an image that isn't similar to ANY other image
in the set, not just one that's far from the mean.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

import numpy as np

log = logging.getLogger("starBoard.dl.outlier_detection")

# MAD to standard deviation conversion factor (for normal distribution)
MAD_SCALE = 1.4826


@dataclass
class OutlierResult:
    """Result of outlier detection for an identity."""
    id_str: str
    n_total: int
    n_inliers: int
    n_outliers: int
    outlier_indices: List[int]
    outlier_paths: List[str]
    inlier_embedding: np.ndarray  # Aggregated embedding from inliers only
    mean_distance: float  # Mean distance from centroid (for diagnostics)
    max_distance: float   # Max distance from centroid (for diagnostics)


def compute_centroid(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute L2-normalized centroid of embeddings.
    
    Args:
        embeddings: (N, D) array of L2-normalized embeddings
        
    Returns:
        (D,) L2-normalized centroid
    """
    centroid = embeddings.mean(axis=0)
    return centroid / (np.linalg.norm(centroid) + 1e-8)


def detect_outliers(
    embeddings: np.ndarray,
    image_paths: List[str],
    id_str: str,
    mad_threshold: float = 3.0,
    min_inliers: int = 1,
    max_outlier_fraction: float = 0.25
) -> OutlierResult:
    """
    Detect outliers using nearest-neighbor similarity with robust statistics.
    
    This method is:
    - Multi-modal friendly: Uses NN similarity, so front/back views of same
      animal stay connected to their respective clusters
    - Robust to contamination: Uses median/MAD instead of mean/std
    - Model-agnostic: Uses relative thresholds, works with triplet/circle loss
    
    Algorithm:
    1. Compute pairwise similarities between all embeddings
    2. For each embedding, find its nearest neighbor similarity
    3. Compute median and MAD of NN similarities
    4. Flag embeddings whose NN similarity is unusually low (isolated points)
    5. Aggregate inlier embeddings into final centroid
    
    Args:
        embeddings: (N, D) array of L2-normalized embeddings
        image_paths: List of image paths corresponding to embeddings
        id_str: Identity ID for logging
        mad_threshold: Number of MAD units below median to consider outlier.
                       3.0 ≈ 3 standard deviations for normal data. (default 3.0)
        min_inliers: Minimum number of inliers to keep (won't exclude if below)
        max_outlier_fraction: Maximum fraction of images to reject (default 0.25)
        
    Returns:
        OutlierResult with inlier embedding and outlier information
    """
    n_total = len(embeddings)
    embed_dim = embeddings.shape[1] if len(embeddings.shape) > 1 else 512
    
    # === Edge cases ===
    
    if n_total == 0:
        return OutlierResult(
            id_str=id_str,
            n_total=0,
            n_inliers=0,
            n_outliers=0,
            outlier_indices=[],
            outlier_paths=[],
            inlier_embedding=np.zeros(embed_dim),
            mean_distance=0.0,
            max_distance=0.0
        )
    
    if n_total == 1:
        return OutlierResult(
            id_str=id_str,
            n_total=1,
            n_inliers=1,
            n_outliers=0,
            outlier_indices=[],
            outlier_paths=[],
            inlier_embedding=embeddings[0],
            mean_distance=0.0,
            max_distance=0.0
        )
    
    if n_total == 2:
        # Two samples - can't do meaningful outlier detection
        # Both images only have each other as neighbors
        centroid = compute_centroid(embeddings)
        distances = 1.0 - (embeddings @ centroid)
        return OutlierResult(
            id_str=id_str,
            n_total=2,
            n_inliers=2,
            n_outliers=0,
            outlier_indices=[],
            outlier_paths=[],
            inlier_embedding=centroid,
            mean_distance=float(distances.mean()),
            max_distance=float(distances.max())
        )
    
    if n_total == 3:
        # Three samples - marginal for statistics, be very conservative
        # Only flag if one image is clearly disconnected from both others
        centroid = compute_centroid(embeddings)
        distances = 1.0 - (embeddings @ centroid)
        
        # Compute pairwise similarities
        similarities = embeddings @ embeddings.T
        np.fill_diagonal(similarities, -np.inf)
        nn_sims = similarities.max(axis=1)
        
        # Only flag if one is dramatically worse than others
        # Use a simple check: worst NN sim < 0.5 * median NN sim
        median_nn = np.median(nn_sims)
        worst_idx = int(np.argmin(nn_sims))
        
        if nn_sims[worst_idx] < 0.5 * median_nn and min_inliers <= 2:
            inlier_mask = np.ones(n_total, dtype=bool)
            inlier_mask[worst_idx] = False
            final_centroid = compute_centroid(embeddings[inlier_mask])
            final_distances = 1.0 - (embeddings @ final_centroid)
            
            log.info("Outlier for %s: 1/3 (NN sim %.3f << median %.3f)",
                     id_str, nn_sims[worst_idx], median_nn)
            
            return OutlierResult(
                id_str=id_str,
                n_total=3,
                n_inliers=2,
                n_outliers=1,
                outlier_indices=[worst_idx],
                outlier_paths=[image_paths[worst_idx]],
                inlier_embedding=final_centroid,
                mean_distance=float(final_distances[inlier_mask].mean()),
                max_distance=float(final_distances.max())
            )
        else:
            return OutlierResult(
                id_str=id_str,
                n_total=3,
                n_inliers=3,
                n_outliers=0,
                outlier_indices=[],
                outlier_paths=[],
                inlier_embedding=centroid,
                mean_distance=float(distances.mean()),
                max_distance=float(distances.max())
            )
    
    # === Main algorithm for n >= 4 ===
    
    # Step 1: Compute pairwise similarities
    similarities = embeddings @ embeddings.T  # (N, N) cosine similarities
    
    # Step 2: Find nearest neighbor similarity for each embedding (excluding self)
    np.fill_diagonal(similarities, -np.inf)
    nn_similarities = similarities.max(axis=1)  # Best match for each image
    
    # Step 3: Robust statistics on NN similarities
    median_nn = np.median(nn_similarities)
    mad = np.median(np.abs(nn_similarities - median_nn))
    
    if mad < 1e-6:
        # All NN similarities nearly identical - no outliers detectable
        centroid = compute_centroid(embeddings)
        distances = 1.0 - (embeddings @ centroid)
        return OutlierResult(
            id_str=id_str,
            n_total=n_total,
            n_inliers=n_total,
            n_outliers=0,
            outlier_indices=[],
            outlier_paths=[],
            inlier_embedding=centroid,
            mean_distance=float(distances.mean()),
            max_distance=float(distances.max())
        )
    
    # Step 4: Identify outliers - images with unusually LOW NN similarity
    # (they're not close to anything)
    # Convert MAD to pseudo-standard-deviation for interpretability
    robust_std = mad * MAD_SCALE
    z_scores = (median_nn - nn_similarities) / robust_std  # Higher = more isolated
    
    inlier_mask = z_scores <= mad_threshold
    
    # Step 5: Apply constraints
    
    # Ensure min_inliers
    if inlier_mask.sum() < min_inliers:
        # Keep the least isolated ones
        sorted_indices = np.argsort(z_scores)[:min_inliers]
        inlier_mask = np.zeros(n_total, dtype=bool)
        inlier_mask[sorted_indices] = True
    
    # Enforce max_outlier_fraction
    max_outliers = int(n_total * max_outlier_fraction)
    n_outliers_proposed = n_total - inlier_mask.sum()
    
    if n_outliers_proposed > max_outliers:
        # Only keep the most extreme outliers
        outlier_indices_all = np.where(~inlier_mask)[0]
        outlier_z_scores = z_scores[outlier_indices_all]
        keep_as_outlier = outlier_indices_all[np.argsort(outlier_z_scores)[-max_outliers:]]
        
        inlier_mask = np.ones(n_total, dtype=bool)
        inlier_mask[keep_as_outlier] = False
    
    # Step 6: Compute final centroid from inliers
    final_centroid = compute_centroid(embeddings[inlier_mask])
    final_distances = 1.0 - (embeddings @ final_centroid)
    
    # Build result
    outlier_indices = list(np.where(~inlier_mask)[0])
    outlier_paths = [image_paths[i] for i in outlier_indices]
    n_inliers = int(inlier_mask.sum())
    n_outliers = n_total - n_inliers
    
    if n_outliers > 0:
        log.info("Outliers for %s: %d/%d (NN-based, MAD threshold=%.1f)",
                 id_str, n_outliers, n_total, mad_threshold)
        for i, (idx, path) in enumerate(zip(outlier_indices, outlier_paths)):
            log.debug("  Outlier %d: %s (NN_sim=%.3f, z=%.2f)",
                      i + 1, path, nn_similarities[idx], z_scores[idx])
    
    return OutlierResult(
        id_str=id_str,
        n_total=n_total,
        n_inliers=n_inliers,
        n_outliers=n_outliers,
        outlier_indices=outlier_indices,
        outlier_paths=outlier_paths,
        inlier_embedding=final_centroid,
        mean_distance=float(final_distances[inlier_mask].mean()) if n_inliers > 0 else 0.0,
        max_distance=float(final_distances.max())
    )


def detect_outliers_legacy(
    embeddings: np.ndarray,
    image_paths: List[str],
    id_str: str,
    z_threshold: float = 2.5,
    min_inliers: int = 1
) -> OutlierResult:
    """
    Legacy outlier detection using centroid-based z-scores.
    
    DEPRECATED: Use detect_outliers() instead. This method has issues with:
    - Multi-modal data (punishes legitimate view variation)
    - Outlier contamination (mean/std are not robust)
    
    Kept for backward compatibility and A/B testing.
    
    Args:
        embeddings: (N, D) array of L2-normalized embeddings
        image_paths: List of image paths corresponding to embeddings
        id_str: Identity ID for logging
        z_threshold: Number of standard deviations to consider outlier (default 2.5)
        min_inliers: Minimum number of inliers to keep (won't exclude if below)
        
    Returns:
        OutlierResult with inlier embedding and outlier information
    """
    n_total = len(embeddings)
    embed_dim = embeddings.shape[1] if len(embeddings.shape) > 1 else 512
    
    # Handle edge cases
    if n_total == 0:
        return OutlierResult(
            id_str=id_str,
            n_total=0,
            n_inliers=0,
            n_outliers=0,
            outlier_indices=[],
            outlier_paths=[],
            inlier_embedding=np.zeros(embed_dim),
            mean_distance=0.0,
            max_distance=0.0
        )
    
    if n_total == 1:
        return OutlierResult(
            id_str=id_str,
            n_total=1,
            n_inliers=1,
            n_outliers=0,
            outlier_indices=[],
            outlier_paths=[],
            inlier_embedding=embeddings[0],
            mean_distance=0.0,
            max_distance=0.0
        )
    
    if n_total == 2:
        # Two samples - not enough for meaningful statistics
        centroid = compute_centroid(embeddings)
        distances = 1.0 - (embeddings @ centroid)
        return OutlierResult(
            id_str=id_str,
            n_total=2,
            n_inliers=2,
            n_outliers=0,
            outlier_indices=[],
            outlier_paths=[],
            inlier_embedding=centroid,
            mean_distance=float(distances.mean()),
            max_distance=float(distances.max())
        )
    
    # Compute centroid and distances
    centroid = compute_centroid(embeddings)
    distances = 1.0 - (embeddings @ centroid)  # cosine distance
    
    # Z-score based detection
    mean_dist = distances.mean()
    std_dist = distances.std()
    
    if std_dist < 1e-6:
        # All embeddings nearly identical - no outliers
        inlier_mask = np.ones(n_total, dtype=bool)
    else:
        z_scores = (distances - mean_dist) / std_dist
        inlier_mask = z_scores <= z_threshold
    
    # Ensure min_inliers
    if inlier_mask.sum() < min_inliers:
        sorted_indices = np.argsort(distances)[:min_inliers]
        inlier_mask = np.zeros(n_total, dtype=bool)
        inlier_mask[sorted_indices] = True
    
    # Final centroid from inliers only
    final_centroid = compute_centroid(embeddings[inlier_mask])
    final_distances = 1.0 - (embeddings @ final_centroid)
    
    outlier_indices = list(np.where(~inlier_mask)[0])
    outlier_paths = [image_paths[i] for i in outlier_indices]
    n_inliers = int(inlier_mask.sum())
    n_outliers = n_total - n_inliers
    
    if n_outliers > 0:
        log.info("Outliers for %s: %d/%d (z>%.1f) [LEGACY]",
                 id_str, n_outliers, n_total, z_threshold)
        for i, (idx, path) in enumerate(zip(outlier_indices, outlier_paths)):
            log.debug("  Outlier %d: %s (distance=%.3f, z=%.2f)",
                      i + 1, path, final_distances[idx],
                      (final_distances[idx] - mean_dist) / std_dist if std_dist > 1e-6 else 0)
    
    return OutlierResult(
        id_str=id_str,
        n_total=n_total,
        n_inliers=n_inliers,
        n_outliers=n_outliers,
        outlier_indices=outlier_indices,
        outlier_paths=outlier_paths,
        inlier_embedding=final_centroid,
        mean_distance=float(final_distances[inlier_mask].mean()) if n_inliers > 0 else 0.0,
        max_distance=float(final_distances.max())
    )
