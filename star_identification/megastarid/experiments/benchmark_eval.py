#!/usr/bin/env python
"""
Benchmark script to validate evaluation optimizations before implementing.

Tests:
1. Vectorized vs loop-based metric computation
2. GPU vs CPU similarity computation
3. Correctness verification (results should match)

Run:
    python -m megastarid.experiments.benchmark_eval
"""
import time
import numpy as np
import torch
from typing import Dict, Tuple


# =============================================================================
# CURRENT IMPLEMENTATION (Loop-based)
# =============================================================================

def compute_metrics_loop(
    similarities: np.ndarray,
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
) -> Dict[str, float]:
    """Current loop-based implementation (copied from trainer.py)."""
    all_aps = []
    all_ranks = []
    
    for i in range(len(query_labels)):
        query_label = query_labels[i]
        sims = similarities[i]
        
        # Sort by similarity
        sorted_indices = np.argsort(-sims)
        sorted_labels = gallery_labels[sorted_indices]
        
        matches = sorted_labels == query_label
        
        if matches.sum() == 0:
            continue
        
        # AP
        cumsum = np.cumsum(matches)
        precision = cumsum / (np.arange(len(matches)) + 1)
        ap = (precision * matches).sum() / matches.sum()
        all_aps.append(ap)
        
        # Rank
        first_match = np.where(matches)[0][0]
        all_ranks.append(first_match + 1)
    
    metrics = {
        'mAP': np.mean(all_aps) if all_aps else 0.0,
        'Rank-1': np.mean([1.0 if r <= 1 else 0.0 for r in all_ranks]) if all_ranks else 0.0,
        'Rank-5': np.mean([1.0 if r <= 5 else 0.0 for r in all_ranks]) if all_ranks else 0.0,
        'Rank-10': np.mean([1.0 if r <= 10 else 0.0 for r in all_ranks]) if all_ranks else 0.0,
    }
    
    return metrics


# =============================================================================
# OPTIMIZED IMPLEMENTATION (Vectorized)
# =============================================================================

def compute_metrics_vectorized(
    similarities: np.ndarray,
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
) -> Dict[str, float]:
    """Fully vectorized metric computation."""
    num_query, num_gallery = similarities.shape
    
    # Sort all queries at once: (num_query, num_gallery)
    sorted_indices = np.argsort(-similarities, axis=1)
    
    # Get sorted labels for all queries at once using advanced indexing
    # sorted_labels[i, j] = gallery_labels[sorted_indices[i, j]]
    sorted_labels = gallery_labels[sorted_indices]  # (num_query, num_gallery)
    
    # Match matrix: (num_query, num_gallery)
    matches = (sorted_labels == query_labels[:, None])
    
    # Check which queries have at least one match
    has_match = matches.any(axis=1)
    
    if not has_match.any():
        return {'mAP': 0.0, 'Rank-1': 0.0, 'Rank-5': 0.0, 'Rank-10': 0.0}
    
    # ===== CMC (Cumulative Matching Characteristics) =====
    # First match position for each query
    # argmax returns first True position when applied to bool array
    first_match_pos = np.argmax(matches, axis=1)  # 0-indexed
    ranks = first_match_pos + 1  # Convert to 1-indexed
    
    # For queries without matches, argmax returns 0, but we filter them out
    valid_ranks = ranks[has_match]
    
    rank1 = float((valid_ranks <= 1).mean())
    rank5 = float((valid_ranks <= 5).mean())
    rank10 = float((valid_ranks <= 10).mean())
    
    # ===== mAP (Mean Average Precision) =====
    # Cumulative matches at each position
    cumsum = np.cumsum(matches, axis=1).astype(np.float64)  # (num_query, num_gallery)
    
    # Precision at each position: cumsum / position
    positions = np.arange(1, num_gallery + 1, dtype=np.float64)  # [1, 2, 3, ...]
    precision_at_k = cumsum / positions  # (num_query, num_gallery)
    
    # AP = sum(precision_at_k * is_match) / num_matches
    # Only count precision at positions where there's a match
    num_matches = matches.sum(axis=1).astype(np.float64)  # (num_query,)
    
    # Avoid division by zero
    num_matches_safe = np.maximum(num_matches, 1)
    
    ap_per_query = (precision_at_k * matches).sum(axis=1) / num_matches_safe
    
    # Only average over queries that have matches
    mAP = float(ap_per_query[has_match].mean())
    
    return {'mAP': mAP, 'Rank-1': rank1, 'Rank-5': rank5, 'Rank-10': rank10}


def compute_metrics_optimized(
    similarities: np.ndarray,
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
) -> Dict[str, float]:
    """
    Optimized metric computation.
    
    Key insight: We don't need full sort for CMC@k, only for mAP.
    - For CMC: Use argpartition (O(n)) to find top-k, then check matches
    - For mAP: Still need argsort, but can process in chunks for memory efficiency
    
    This version also processes queries in batches to improve cache efficiency.
    """
    num_query, num_gallery = similarities.shape
    
    # Pre-compute match matrix (before sorting) for CMC optimization
    # matches_unsorted[i,j] = True if query_labels[i] == gallery_labels[j]
    matches_unsorted = (query_labels[:, None] == gallery_labels[None, :])
    
    # Check which queries have at least one match
    has_match = matches_unsorted.any(axis=1)
    
    if not has_match.any():
        return {'mAP': 0.0, 'Rank-1': 0.0, 'Rank-5': 0.0, 'Rank-10': 0.0}
    
    # ===== CMC using argpartition (faster for small k) =====
    # For Rank-k, we only need to check top-k items
    max_k = 10
    
    # argpartition is O(n) vs argsort O(n log n)
    # Get indices of top-k highest similarities
    if num_gallery > max_k:
        # Partition: items at indices [:max_k] are the max_k largest (unordered)
        top_k_indices = np.argpartition(-similarities, max_k, axis=1)[:, :max_k]
    else:
        top_k_indices = np.argsort(-similarities, axis=1)[:, :max_k]
    
    # Check if any of top-k are matches
    # Gather the match status for top-k indices
    row_idx = np.arange(num_query)[:, None]
    top_k_matches = matches_unsorted[row_idx, top_k_indices]  # (num_query, max_k)
    
    # For proper rank computation, we need similarity-sorted order within top-k
    top_k_sims = similarities[row_idx, top_k_indices]
    top_k_order = np.argsort(-top_k_sims, axis=1)  # Sort within top-k
    top_k_matches_sorted = np.take_along_axis(top_k_matches, top_k_order, axis=1)
    
    # First match position within top-k (0-indexed)
    has_match_in_topk = top_k_matches_sorted.any(axis=1)
    first_match_topk = np.argmax(top_k_matches_sorted, axis=1)
    
    # For queries with match in top-k, rank = position + 1
    # For queries without match in top-k, we need full sort (rare case)
    ranks = np.full(num_query, num_gallery + 1, dtype=np.int32)  # Default: no match
    ranks[has_match_in_topk] = first_match_topk[has_match_in_topk] + 1
    
    # Handle queries that have matches but not in top-k (need full sort for these)
    need_full_sort = has_match & ~has_match_in_topk
    if need_full_sort.any():
        for i in np.where(need_full_sort)[0]:
            sorted_idx = np.argsort(-similarities[i])
            sorted_matches = matches_unsorted[i, sorted_idx]
            ranks[i] = np.argmax(sorted_matches) + 1
    
    valid_ranks = ranks[has_match]
    rank1 = float((valid_ranks <= 1).mean())
    rank5 = float((valid_ranks <= 5).mean())
    rank10 = float((valid_ranks <= 10).mean())
    
    # ===== mAP: Still need full sort =====
    # Process in chunks to be more cache-friendly
    chunk_size = 256
    all_aps = []
    
    for start in range(0, num_query, chunk_size):
        end = min(start + chunk_size, num_query)
        chunk_sims = similarities[start:end]
        chunk_matches = matches_unsorted[start:end]
        chunk_has_match = has_match[start:end]
        
        # Sort this chunk
        sorted_indices = np.argsort(-chunk_sims, axis=1)
        
        # Gather sorted matches
        row_idx = np.arange(end - start)[:, None]
        sorted_matches = chunk_matches[row_idx, sorted_indices]
        
        # Compute AP for each query in chunk
        cumsum = np.cumsum(sorted_matches, axis=1).astype(np.float64)
        positions = np.arange(1, num_gallery + 1, dtype=np.float64)
        precision_at_k = cumsum / positions
        
        num_matches = sorted_matches.sum(axis=1).astype(np.float64)
        num_matches_safe = np.maximum(num_matches, 1)
        
        ap_chunk = (precision_at_k * sorted_matches).sum(axis=1) / num_matches_safe
        all_aps.extend(ap_chunk[chunk_has_match].tolist())
    
    mAP = float(np.mean(all_aps)) if all_aps else 0.0
    
    return {'mAP': mAP, 'Rank-1': rank1, 'Rank-5': rank5, 'Rank-10': rank10}


# =============================================================================
# SIMILARITY COMPUTATION BENCHMARKS
# =============================================================================

def compute_similarity_cpu(
    query_emb: np.ndarray,
    gallery_emb: np.ndarray,
) -> np.ndarray:
    """CPU numpy matmul."""
    return query_emb @ gallery_emb.T


def compute_similarity_gpu(
    query_emb: torch.Tensor,
    gallery_emb: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    """GPU torch matmul, return numpy."""
    with torch.no_grad():
        query_gpu = query_emb.to(device)
        gallery_gpu = gallery_emb.to(device)
        sim = (query_gpu @ gallery_gpu.T).cpu().numpy()
    return sim


# =============================================================================
# BENCHMARK FUNCTIONS
# =============================================================================

def generate_test_data(
    num_query: int,
    num_gallery: int,
    embedding_dim: int,
    num_identities: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate realistic test data."""
    np.random.seed(42)
    
    # Random embeddings (L2 normalized)
    query_emb = np.random.randn(num_query, embedding_dim).astype(np.float32)
    query_emb = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)
    
    gallery_emb = np.random.randn(num_gallery, embedding_dim).astype(np.float32)
    gallery_emb = gallery_emb / np.linalg.norm(gallery_emb, axis=1, keepdims=True)
    
    # Random labels (ensure each query has at least one match in gallery)
    query_labels = np.random.randint(0, num_identities, size=num_query)
    gallery_labels = np.random.randint(0, num_identities, size=num_gallery)
    
    # Ensure some matches exist
    gallery_labels[:num_query] = query_labels  # First num_query gallery items match queries
    np.random.shuffle(gallery_labels)
    
    return query_emb, gallery_emb, query_labels, gallery_labels


def benchmark_metrics(
    similarities: np.ndarray,
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
    num_runs: int = 5,
) -> None:
    """Benchmark loop vs vectorized vs optimized metrics computation."""
    print("\n" + "=" * 60)
    print("BENCHMARK: Metric Computation")
    print("=" * 60)
    print(f"Query size: {len(query_labels)}, Gallery size: {len(gallery_labels)}")
    
    # Warmup
    _ = compute_metrics_loop(similarities, query_labels, gallery_labels)
    _ = compute_metrics_vectorized(similarities, query_labels, gallery_labels)
    _ = compute_metrics_optimized(similarities, query_labels, gallery_labels)
    
    # Benchmark loop version
    times_loop = []
    for _ in range(num_runs):
        start = time.perf_counter()
        result_loop = compute_metrics_loop(similarities, query_labels, gallery_labels)
        times_loop.append(time.perf_counter() - start)
    
    # Benchmark vectorized version
    times_vec = []
    for _ in range(num_runs):
        start = time.perf_counter()
        result_vec = compute_metrics_vectorized(similarities, query_labels, gallery_labels)
        times_vec.append(time.perf_counter() - start)
    
    # Benchmark optimized version
    times_opt = []
    for _ in range(num_runs):
        start = time.perf_counter()
        result_opt = compute_metrics_optimized(similarities, query_labels, gallery_labels)
        times_opt.append(time.perf_counter() - start)
    
    avg_loop = np.mean(times_loop)
    avg_vec = np.mean(times_vec)
    avg_opt = np.mean(times_opt)
    speedup_vec = avg_loop / avg_vec
    speedup_opt = avg_loop / avg_opt
    
    print(f"\nLoop-based:  {avg_loop*1000:.2f} ms (±{np.std(times_loop)*1000:.2f})")
    print(f"Vectorized:  {avg_vec*1000:.2f} ms (±{np.std(times_vec)*1000:.2f}) -> {speedup_vec:.1f}x")
    print(f"Optimized:   {avg_opt*1000:.2f} ms (±{np.std(times_opt)*1000:.2f}) -> {speedup_opt:.1f}x")
    
    # Verify correctness
    print("\n--- Correctness Check ---")
    print(f"Loop:       mAP={result_loop['mAP']:.6f}, R1={result_loop['Rank-1']:.6f}, R5={result_loop['Rank-5']:.6f}")
    print(f"Vectorized: mAP={result_vec['mAP']:.6f}, R1={result_vec['Rank-1']:.6f}, R5={result_vec['Rank-5']:.6f}")
    print(f"Optimized:  mAP={result_opt['mAP']:.6f}, R1={result_opt['Rank-1']:.6f}, R5={result_opt['Rank-5']:.6f}")
    
    # Check if results match
    all_match = True
    for key in ['mAP', 'Rank-1', 'Rank-5', 'Rank-10']:
        diff_vec = abs(result_loop[key] - result_vec[key])
        diff_opt = abs(result_loop[key] - result_opt[key])
        if diff_vec > 1e-6:
            print(f"  ⚠️  Vectorized {key} differs by {diff_vec:.8f}")
            all_match = False
        if diff_opt > 1e-6:
            print(f"  ⚠️  Optimized {key} differs by {diff_opt:.8f}")
            all_match = False
    
    if all_match:
        print("  ✅ All results match perfectly!")
    else:
        print("  ❌ Results differ - need to investigate!")
    
    return max(speedup_vec, speedup_opt), all_match


def benchmark_similarity(
    query_emb: np.ndarray,
    gallery_emb: np.ndarray,
    device: torch.device,
    num_runs: int = 5,
) -> None:
    """Benchmark CPU vs GPU similarity computation."""
    print("\n" + "=" * 60)
    print("BENCHMARK: Similarity Computation")
    print("=" * 60)
    print(f"Query: {query_emb.shape}, Gallery: {gallery_emb.shape}")
    
    # Convert to torch
    query_torch = torch.from_numpy(query_emb)
    gallery_torch = torch.from_numpy(gallery_emb)
    
    # Warmup
    _ = compute_similarity_cpu(query_emb, gallery_emb)
    if device.type == 'cuda':
        _ = compute_similarity_gpu(query_torch, gallery_torch, device)
        torch.cuda.synchronize()
    
    # Benchmark CPU
    times_cpu = []
    for _ in range(num_runs):
        start = time.perf_counter()
        result_cpu = compute_similarity_cpu(query_emb, gallery_emb)
        times_cpu.append(time.perf_counter() - start)
    
    # Benchmark GPU
    if device.type == 'cuda':
        times_gpu = []
        for _ in range(num_runs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            result_gpu = compute_similarity_gpu(query_torch, gallery_torch, device)
            torch.cuda.synchronize()
            times_gpu.append(time.perf_counter() - start)
        
        avg_cpu = np.mean(times_cpu)
        avg_gpu = np.mean(times_gpu)
        speedup = avg_cpu / avg_gpu
        
        print(f"\nCPU numpy:   {avg_cpu*1000:.2f} ms (±{np.std(times_cpu)*1000:.2f})")
        print(f"GPU torch:   {avg_gpu*1000:.2f} ms (±{np.std(times_gpu)*1000:.2f})")
        print(f"Speedup:     {speedup:.1f}x")
        
        # Verify correctness
        max_diff = np.abs(result_cpu - result_gpu).max()
        print(f"\nMax difference: {max_diff:.8f}")
        if max_diff < 1e-5:
            print("  ✅ Results match (within fp32 tolerance)!")
        else:
            print("  ⚠️  Larger difference - but likely just fp32 precision")
    else:
        print("\nNo CUDA available - skipping GPU benchmark")


def run_all_benchmarks():
    """Run all benchmarks with different data sizes."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Test configurations (simulating real star_dataset sizes)
    configs = [
        # (num_query, num_gallery, embedding_dim, num_identities, description)
        (100, 500, 512, 50, "Small (quick test)"),
        (500, 2000, 512, 100, "Medium (typical validation)"),
        (1000, 5000, 512, 200, "Large (full dataset)"),
        (2000, 10000, 512, 300, "Very large (stress test)"),
    ]
    
    results = []
    
    for num_q, num_g, dim, n_ids, desc in configs:
        print(f"\n\n{'#'*60}")
        print(f"# CONFIG: {desc}")
        print(f"# Queries: {num_q}, Gallery: {num_g}, Dim: {dim}, IDs: {n_ids}")
        print(f"{'#'*60}")
        
        # Generate data
        query_emb, gallery_emb, query_labels, gallery_labels = generate_test_data(
            num_q, num_g, dim, n_ids
        )
        
        # First compute similarities (needed for metric benchmark)
        similarities = query_emb @ gallery_emb.T
        
        # Benchmark metrics
        speedup, correct = benchmark_metrics(
            similarities, query_labels, gallery_labels
        )
        
        # Benchmark similarity
        benchmark_similarity(query_emb, gallery_emb, device)
        
        results.append({
            'config': desc,
            'num_query': num_q,
            'num_gallery': num_g,
            'metric_speedup': speedup,
            'correct': correct,
        })
    
    # Summary
    print("\n\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Config':<30} {'Queries':>8} {'Gallery':>8} {'Speedup':>10} {'Correct':>8}")
    print("-" * 60)
    for r in results:
        print(f"{r['config']:<30} {r['num_query']:>8} {r['num_gallery']:>8} {r['metric_speedup']:>9.1f}x {'✅' if r['correct'] else '❌':>8}")
    
    all_correct = all(r['correct'] for r in results)
    avg_speedup = np.mean([r['metric_speedup'] for r in results])
    
    print("-" * 60)
    print(f"Average metric speedup: {avg_speedup:.1f}x")
    print(f"All results correct: {'✅ YES' if all_correct else '❌ NO'}")
    
    if all_correct:
        print("\n🎉 Validation PASSED - safe to implement optimizations!")
    else:
        print("\n⚠️  Validation FAILED - need to fix vectorized implementation!")
    
    return all_correct


if __name__ == '__main__':
    success = run_all_benchmarks()
    exit(0 if success else 1)

