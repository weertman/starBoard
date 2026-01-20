"""
Validation tests for proposed optimizations to evaluation/inference code.

Tests:
1. Vectorized Jaccard distance vs original nested loop
2. autocast context manager that works on CPU
3. Vectorized k-reciprocal encoding vs original loop

Each test validates:
- Correctness: outputs match original implementation
- Speed: improved version is faster
"""

import numpy as np
import torch
import time
from contextlib import nullcontext
from typing import Tuple
from torch.amp import autocast


# =============================================================================
# Test 1: Jaccard Distance Vectorization
# =============================================================================

def jaccard_original(V: np.ndarray, num_query: int, num_gallery: int) -> np.ndarray:
    """Original nested loop implementation."""
    jaccard_dist = np.zeros((num_query, num_gallery), dtype=np.float32)
    
    for i in range(num_query):
        for j in range(num_gallery):
            g_idx = num_query + j
            minimum = np.minimum(V[i], V[g_idx])
            maximum = np.maximum(V[i], V[g_idx])
            jaccard_dist[i, j] = 1 - np.sum(minimum) / (np.sum(maximum) + 1e-8)
    
    return jaccard_dist


def jaccard_vectorized(V: np.ndarray, num_query: int, num_gallery: int) -> np.ndarray:
    """Vectorized implementation using broadcasting."""
    V_query = V[:num_query]  # (num_query, num_all)
    V_gallery = V[num_query:num_query + num_gallery]  # (num_gallery, num_all)
    
    # Expand for broadcasting: (num_query, 1, num_all) vs (1, num_gallery, num_all)
    # This computes all pairwise min/max in one go
    minimum = np.minimum(V_query[:, None, :], V_gallery[None, :, :])
    maximum = np.maximum(V_query[:, None, :], V_gallery[None, :, :])
    
    # Sum over feature dimension and compute Jaccard
    min_sum = minimum.sum(axis=2)  # (num_query, num_gallery)
    max_sum = maximum.sum(axis=2)  # (num_query, num_gallery)
    
    jaccard_dist = 1 - min_sum / (max_sum + 1e-8)
    
    return jaccard_dist.astype(np.float32)


def jaccard_vectorized_chunked(V: np.ndarray, num_query: int, num_gallery: int, 
                                chunk_size: int = 100) -> np.ndarray:
    """
    Chunked vectorized implementation for memory efficiency.
    Useful when num_query * num_gallery * num_all would exceed memory.
    """
    V_query = V[:num_query]
    V_gallery = V[num_query:num_query + num_gallery]
    
    jaccard_dist = np.zeros((num_query, num_gallery), dtype=np.float32)
    
    # Process in chunks to avoid memory blowup
    for i in range(0, num_query, chunk_size):
        i_end = min(i + chunk_size, num_query)
        V_q_chunk = V_query[i:i_end]  # (chunk, num_all)
        
        minimum = np.minimum(V_q_chunk[:, None, :], V_gallery[None, :, :])
        maximum = np.maximum(V_q_chunk[:, None, :], V_gallery[None, :, :])
        
        min_sum = minimum.sum(axis=2)
        max_sum = maximum.sum(axis=2)
        
        jaccard_dist[i:i_end] = 1 - min_sum / (max_sum + 1e-8)
    
    return jaccard_dist


def test_jaccard_vectorization():
    """Test that vectorized Jaccard matches original and is faster."""
    print("\n" + "=" * 60)
    print("TEST 1: Jaccard Distance Vectorization")
    print("=" * 60)
    
    # Test with realistic sizes
    test_cases = [
        (50, 100, 150),    # Small: 50 query, 100 gallery, 150 total features
        (100, 500, 600),   # Medium
        (200, 1000, 1200), # Larger
    ]
    
    for num_query, num_gallery, num_all in test_cases:
        print(f"\n--- num_query={num_query}, num_gallery={num_gallery}, num_all={num_all} ---")
        
        # Create random V matrix (sparse-ish, like real k-reciprocal encoding)
        np.random.seed(42)
        V = np.random.rand(num_all, num_all).astype(np.float32)
        V = V * (np.random.rand(num_all, num_all) > 0.7)  # Make ~70% zeros
        V = V / (V.sum(axis=1, keepdims=True) + 1e-8)  # Normalize rows
        
        # Original implementation
        start = time.perf_counter()
        result_original = jaccard_original(V, num_query, num_gallery)
        time_original = time.perf_counter() - start
        
        # Vectorized implementation
        start = time.perf_counter()
        result_vectorized = jaccard_vectorized(V, num_query, num_gallery)
        time_vectorized = time.perf_counter() - start
        
        # Chunked vectorized (for comparison)
        start = time.perf_counter()
        result_chunked = jaccard_vectorized_chunked(V, num_query, num_gallery)
        time_chunked = time.perf_counter() - start
        
        # Validate correctness
        max_diff_vec = np.abs(result_original - result_vectorized).max()
        max_diff_chunk = np.abs(result_original - result_chunked).max()
        
        print(f"  Original:   {time_original:.4f}s")
        print(f"  Vectorized: {time_vectorized:.4f}s ({time_original/time_vectorized:.1f}x speedup)")
        print(f"  Chunked:    {time_chunked:.4f}s ({time_original/time_chunked:.1f}x speedup)")
        print(f"  Max diff (vectorized): {max_diff_vec:.2e}")
        print(f"  Max diff (chunked):    {max_diff_chunk:.2e}")
        
        # Assert correctness (allow small numerical differences)
        assert max_diff_vec < 1e-5, f"Vectorized result differs too much: {max_diff_vec}"
        assert max_diff_chunk < 1e-5, f"Chunked result differs too much: {max_diff_chunk}"
        assert time_vectorized < time_original, "Vectorized should be faster!"
        
        print("  ✓ PASSED")
    
    print("\n✓ All Jaccard vectorization tests passed!")


# =============================================================================
# Test 2: Device-aware autocast
# =============================================================================

def get_autocast_context(device: torch.device):
    """Get appropriate autocast context for device."""
    if device.type == 'cuda':
        return autocast('cuda')
    else:
        return nullcontext()


def dummy_inference_original(images: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Original approach - always uses autocast('cuda')."""
    # This will fail or warn on CPU
    with torch.no_grad(), autocast('cuda'):
        result = images.mean(dim=(2, 3))  # Simple operation
    return result


def dummy_inference_fixed(images: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Fixed approach - device-aware autocast."""
    amp_ctx = get_autocast_context(device)
    with torch.no_grad(), amp_ctx:
        result = images.mean(dim=(2, 3))
    return result


def test_autocast_cpu_fix():
    """Test that fixed autocast works on both CPU and CUDA."""
    print("\n" + "=" * 60)
    print("TEST 2: Device-aware autocast")
    print("=" * 60)
    
    # Test on CPU
    print("\n--- Testing on CPU ---")
    device_cpu = torch.device('cpu')
    images_cpu = torch.randn(4, 3, 224, 224, device=device_cpu)
    
    # Fixed version should work fine
    try:
        result_fixed = dummy_inference_fixed(images_cpu, device_cpu)
        print(f"  Fixed version on CPU: ✓ (shape={result_fixed.shape})")
    except Exception as e:
        print(f"  Fixed version on CPU: ✗ FAILED - {e}")
        raise
    
    # Original version - check if it causes issues
    try:
        # autocast('cuda') on CPU may work but is semantically wrong
        result_original = dummy_inference_original(images_cpu, device_cpu)
        print(f"  Original version on CPU: ran without error (but semantically wrong)")
    except Exception as e:
        print(f"  Original version on CPU: raised error (as expected on some PyTorch versions)")
    
    # Test on CUDA if available
    if torch.cuda.is_available():
        print("\n--- Testing on CUDA ---")
        device_cuda = torch.device('cuda')
        images_cuda = torch.randn(4, 3, 224, 224, device=device_cuda)
        
        result_fixed = dummy_inference_fixed(images_cuda, device_cuda)
        result_original = dummy_inference_original(images_cuda, device_cuda)
        
        # Results should be identical (or very close due to fp16 vs fp32)
        diff = (result_fixed - result_original).abs().max().item()
        print(f"  Fixed version on CUDA: ✓")
        print(f"  Original version on CUDA: ✓")
        print(f"  Max difference: {diff:.2e}")
        
        assert diff < 1e-3, f"Results differ too much: {diff}"
    else:
        print("\n--- CUDA not available, skipping CUDA test ---")
    
    print("\n✓ autocast fix tests passed!")


# =============================================================================
# Test 3: k-Reciprocal Encoding Vectorization
# =============================================================================

def build_k_reciprocal_original(
    all_rank: np.ndarray,
    original_dist: np.ndarray,
    k1: int = 20,
) -> np.ndarray:
    """Original nested loop implementation for k-reciprocal encoding."""
    num_all = all_rank.shape[0]
    V = np.zeros((num_all, num_all), dtype=np.float32)
    
    for i in range(num_all):
        # Forward k-nearest neighbors
        forward_k_neighbors = all_rank[i, :k1 + 1]
        
        # k-reciprocal neighbors: mutual nearest neighbors
        k_reciprocal_index = []
        for candidate in forward_k_neighbors:
            candidate_forward_k = all_rank[candidate, :k1 + 1]
            if i in candidate_forward_k:
                k_reciprocal_index.append(candidate)
        k_reciprocal_index = np.array(k_reciprocal_index)
        
        # Expand k-reciprocal neighbors
        k_reciprocal_expansion_index = k_reciprocal_index.copy()
        for candidate in k_reciprocal_index:
            candidate_forward_k = all_rank[candidate, :int(k1 / 2) + 1]
            candidate_reciprocal = []
            for c in candidate_forward_k:
                c_forward_k = all_rank[c, :int(k1 / 2) + 1]
                if candidate in c_forward_k:
                    candidate_reciprocal.append(c)
            
            if len(candidate_reciprocal) > 2 / 3 * len(candidate_forward_k):
                k_reciprocal_expansion_index = np.union1d(
                    k_reciprocal_expansion_index, candidate_reciprocal
                )
        
        # Gaussian weighted encoding
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = weight / np.sum(weight)
    
    return V


def build_k_reciprocal_vectorized(
    all_rank: np.ndarray,
    original_dist: np.ndarray,
    k1: int = 20,
) -> np.ndarray:
    """
    Partially vectorized implementation for k-reciprocal encoding.
    
    The core insight: checking "if i in candidate's top-k" can be vectorized
    by building a sparse adjacency matrix of who is in whose top-k.
    """
    num_all = all_rank.shape[0]
    V = np.zeros((num_all, num_all), dtype=np.float32)
    
    # Build "is_in_topk" matrix: is_in_topk[i, j] = True if j is in i's top-k1
    # This is the key vectorization: instead of checking membership in loops,
    # we precompute all memberships at once
    topk_neighbors = all_rank[:, :k1 + 1]  # (num_all, k1+1)
    
    # Create sparse indicator: for each i, mark which j's are in top-k
    # We'll use a dense boolean matrix for simplicity (could use sparse for memory)
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
    
    # Now we can vectorize the reciprocal check
    # k_reciprocal[i, j] = is_in_topk[i, j] AND is_in_topk[j, i]
    is_k_reciprocal = is_in_topk & is_in_topk.T
    
    # For expansion, we still need a loop, but inner operations are vectorized
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
            # Check which of these are reciprocal with candidate
            reciprocal_with_candidate = is_in_topk_half[candidate_half_neighbors, candidate]
            candidate_reciprocal = candidate_half_neighbors[reciprocal_with_candidate]
            
            # Expansion criterion: > 2/3 overlap
            if len(candidate_reciprocal) > 2 / 3 * len(candidate_half_neighbors):
                expansion_set.update(candidate_reciprocal)
        
        k_reciprocal_expansion_index = np.array(list(expansion_set))
        
        # Gaussian weighted encoding
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = weight / (np.sum(weight) + 1e-8)
    
    return V


def test_k_reciprocal_vectorization():
    """Test that vectorized k-reciprocal matches original and is faster."""
    print("\n" + "=" * 60)
    print("TEST 3: k-Reciprocal Encoding Vectorization")
    print("=" * 60)
    
    test_cases = [
        (100, 20),   # 100 samples, k1=20
        (300, 20),   # 300 samples
        (500, 20),   # 500 samples
    ]
    
    for num_all, k1 in test_cases:
        print(f"\n--- num_all={num_all}, k1={k1} ---")
        
        # Create random embeddings and compute distance
        np.random.seed(42)
        features = np.random.randn(num_all, 128).astype(np.float32)
        features = features / np.linalg.norm(features, axis=1, keepdims=True)
        
        original_dist = 1 - np.dot(features, features.T)
        original_dist = np.clip(original_dist, 0, 2)
        all_rank = np.argsort(original_dist, axis=1)
        
        # Original implementation
        start = time.perf_counter()
        V_original = build_k_reciprocal_original(all_rank, original_dist, k1)
        time_original = time.perf_counter() - start
        
        # Vectorized implementation
        start = time.perf_counter()
        V_vectorized = build_k_reciprocal_vectorized(all_rank, original_dist, k1)
        time_vectorized = time.perf_counter() - start
        
        # Validate correctness
        max_diff = np.abs(V_original - V_vectorized).max()
        mean_diff = np.abs(V_original - V_vectorized).mean()
        
        print(f"  Original:   {time_original:.4f}s")
        print(f"  Vectorized: {time_vectorized:.4f}s ({time_original/time_vectorized:.1f}x speedup)")
        print(f"  Max diff: {max_diff:.2e}, Mean diff: {mean_diff:.2e}")
        
        # Check that non-zero pattern matches
        nonzero_orig = V_original > 0
        nonzero_vec = V_vectorized > 0
        pattern_match = (nonzero_orig == nonzero_vec).all()
        print(f"  Non-zero pattern matches: {pattern_match}")
        
        # Assert correctness
        assert max_diff < 1e-5, f"V matrices differ too much: {max_diff}"
        assert pattern_match, "Non-zero patterns don't match!"
        
        print("  ✓ PASSED")
    
    print("\n✓ All k-reciprocal vectorization tests passed!")


# =============================================================================
# Run all tests
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("OPTIMIZATION VALIDATION TESTS")
    print("=" * 60)
    
    test_jaccard_vectorization()
    test_autocast_cpu_fix()
    test_k_reciprocal_vectorization()
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED! Safe to implement optimizations.")
    print("=" * 60)


