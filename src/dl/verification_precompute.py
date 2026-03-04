"""
Verification precomputation for best-photo pairwise comparison.

Computes P(same individual) for all query-gallery pairs using only
the "best" photo from each identity to keep computation manageable.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable

import numpy as np

from src.data.id_registry import list_ids
from src.data.best_photo import load_best_rel_path
from .image_cache import get_cached_images, CACHE_ROOT

log = logging.getLogger("starBoard.dl.verification_precompute")


def get_best_photo_for_id(target: str, id_str: str) -> Optional[str]:
    """
    Get the best photo path (cached version) for an identity.
    
    Uses the best_photo sidecar if set, otherwise falls back to
    the first cached image.
    
    Args:
        target: "Gallery" or "Queries"
        id_str: The identity ID
        
    Returns:
        Path to the cached best photo, or None if no images
    """
    # Get cached images for this identity
    cached = get_cached_images(target, id_str)
    if not cached:
        return None
    
    # Check if best photo is explicitly set
    best_rel = load_best_rel_path(target, id_str)
    
    if best_rel:
        # Find matching cached image by filename stem
        # (cached images are .png, originals may be .jpg/.jpeg etc)
        stem = Path(best_rel).stem
        for cached_path in cached:
            if cached_path.stem == stem:
                return str(cached_path)
    
    # Fallback: first cached image
    return str(cached[0])


def select_best_photos() -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Select best photos for all gallery and query identities.
    
    Returns:
        (gallery_best, query_best) dicts mapping id -> cached_image_path
    """
    gallery_ids = list_ids("Gallery")
    query_ids = list_ids("Queries")
    
    gallery_best: Dict[str, str] = {}
    query_best: Dict[str, str] = {}
    
    for gid in gallery_ids:
        path = get_best_photo_for_id("Gallery", gid)
        if path:
            gallery_best[gid] = path
    
    for qid in query_ids:
        path = get_best_photo_for_id("Queries", qid)
        if path:
            query_best[qid] = path
    
    log.info(
        "Selected best photos: %d gallery, %d queries",
        len(gallery_best), len(query_best)
    )
    
    return gallery_best, query_best


class VerificationPrecomputer:
    """
    Compute pairwise verification scores for best photos.
    
    Uses the VerificationInference wrapper for efficient batch prediction.
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        image_size: int = 224,
    ):
        """
        Initialize the precomputer.
        
        Args:
            checkpoint_path: Path to verification model checkpoint
            device: Device to run inference on
            image_size: Input image size
        """
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.image_size = image_size
        self._inference = None
    
    def _ensure_inference(self):
        """Lazy-load the inference model."""
        if self._inference is None:
            from star_identification.megastar_identity_verification.inference import (
                VerificationInference
            )
            self._inference = VerificationInference(
                self.checkpoint_path,
                device=self.device,
                image_size=self.image_size,
            )
    
    def compute_all_pairs(
        self,
        gallery_best: Dict[str, str],
        query_best: Dict[str, str],
        batch_size: int = 16,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        Compute verification matrix for all (query, gallery) pairs.
        
        Args:
            gallery_best: Dict mapping gallery_id -> best_photo_path
            query_best: Dict mapping query_id -> best_photo_path
            batch_size: Number of pairs per batch
            progress_callback: Optional callback(message, current, total)
            
        Returns:
            (verification_matrix, query_ids, gallery_ids)
            Matrix is shape (n_queries, n_gallery) with P(same) values
        """
        self._ensure_inference()
        
        # Sort IDs for consistent ordering
        query_ids = sorted(query_best.keys())
        gallery_ids = sorted(gallery_best.keys())
        
        n_queries = len(query_ids)
        n_gallery = len(gallery_ids)
        total_pairs = n_queries * n_gallery
        
        if total_pairs == 0:
            return np.array([]).reshape(0, 0), [], []
        
        log.info(
            "Computing verification for %d queries x %d gallery = %d pairs",
            n_queries, n_gallery, total_pairs
        )
        
        # Build all pairs
        pairs: List[Tuple[str, str]] = []
        for qid in query_ids:
            q_path = query_best[qid]
            for gid in gallery_ids:
                g_path = gallery_best[gid]
                pairs.append((q_path, g_path))
        
        # Batch prediction with progress
        start_time = time.time()
        
        def batch_progress(current: int, total: int):
            if progress_callback:
                elapsed = time.time() - start_time
                pairs_per_sec = current / elapsed if elapsed > 0 else 0
                remaining = total - current
                eta = remaining / pairs_per_sec if pairs_per_sec > 0 else 0
                progress_callback(
                    f"Verification: {current}/{total} pairs ({pairs_per_sec:.1f}/s, ETA: {eta:.0f}s)",
                    current,
                    total
                )
        
        probs = self._inference.predict_batch(
            pairs,
            batch_size=batch_size,
            progress_callback=batch_progress,
        )
        
        # Reshape to matrix
        verification_matrix = probs.reshape(n_queries, n_gallery)
        
        elapsed = time.time() - start_time
        log.info(
            "Verification complete: %d pairs in %.1fs (%.1f pairs/sec)",
            total_pairs, elapsed, total_pairs / elapsed
        )
        
        return verification_matrix, query_ids, gallery_ids
    
    def save_results(
        self,
        output_dir: Path,
        verification_matrix: np.ndarray,
        query_ids: List[str],
        gallery_ids: List[str],
        gallery_best: Dict[str, str],
        query_best: Dict[str, str],
    ):
        """
        Save verification results to disk.
        
        Args:
            output_dir: Directory to save results
            verification_matrix: (n_queries, n_gallery) matrix
            query_ids: List of query IDs (row order)
            gallery_ids: List of gallery IDs (column order)
            gallery_best: Best photo paths for gallery
            query_best: Best photo paths for queries
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save verification matrix
        np.savez_compressed(
            output_dir / "verification_scores.npz",
            verification=verification_matrix.astype(np.float32)
        )
        
        # Save ID mapping
        with open(output_dir / "id_mapping.json", 'w', encoding='utf-8') as f:
            json.dump({
                "query_ids": query_ids,
                "gallery_ids": gallery_ids,
            }, f, indent=2)
        
        # Save best photos mapping
        with open(output_dir / "best_photos.json", 'w', encoding='utf-8') as f:
            json.dump({
                "gallery": gallery_best,
                "queries": query_best,
            }, f, indent=2)
        
        # Save metadata
        with open(output_dir / "metadata.json", 'w', encoding='utf-8') as f:
            json.dump({
                "checkpoint_path": str(self.checkpoint_path),
                "device": self.device,
                "image_size": self.image_size,
                "n_queries": len(query_ids),
                "n_gallery": len(gallery_ids),
                "n_pairs": len(query_ids) * len(gallery_ids),
                "computed_at": datetime.utcnow().isoformat() + "Z",
            }, f, indent=2)
        
        log.info("Saved verification results to %s", output_dir)


def run_verification_precompute(
    checkpoint_path: str,
    output_dir: Path,
    device: str = "cuda",
    batch_size: int = 16,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> Tuple[bool, str]:
    """
    Run full verification precomputation pipeline.
    
    Args:
        checkpoint_path: Path to verification model checkpoint
        output_dir: Directory to save results
        device: Device to run inference on
        batch_size: Number of pairs per batch
        progress_callback: Optional callback(message, current, total)
        
    Returns:
        (success, message) tuple
    """
    try:
        # Select best photos
        if progress_callback:
            progress_callback("Selecting best photos...", 0, 100)
        
        gallery_best, query_best = select_best_photos()
        
        if not gallery_best:
            return False, "No gallery photos found (run image cache first)"
        if not query_best:
            return False, "No query photos found (run image cache first)"
        
        # Create precomputer
        precomputer = VerificationPrecomputer(
            checkpoint_path=checkpoint_path,
            device=device,
        )
        
        # Compute all pairs
        verification_matrix, query_ids, gallery_ids = precomputer.compute_all_pairs(
            gallery_best=gallery_best,
            query_best=query_best,
            batch_size=batch_size,
            progress_callback=progress_callback,
        )
        
        # Save results
        if progress_callback:
            progress_callback("Saving results...", 99, 100)
        
        precomputer.save_results(
            output_dir=output_dir,
            verification_matrix=verification_matrix,
            query_ids=query_ids,
            gallery_ids=gallery_ids,
            gallery_best=gallery_best,
            query_best=query_best,
        )
        
        return True, f"Computed {len(query_ids) * len(gallery_ids)} verification pairs"
        
    except Exception as e:
        log.error("Verification precomputation failed: %s", e)
        import traceback
        traceback.print_exc()
        return False, f"Error: {str(e)}"


def _load_existing_verification_artifacts(
    output_dir: Path,
) -> tuple[Optional[np.ndarray], List[str], List[str], Dict[str, Dict[str, str]]]:
    """
    Load existing verification artifacts if available.

    Returns:
        (matrix_or_none, query_ids, gallery_ids, best_photos)
    """
    output_dir = Path(output_dir)
    matrix_path = output_dir / "verification_scores.npz"
    mapping_path = output_dir / "id_mapping.json"
    best_path = output_dir / "best_photos.json"

    matrix: Optional[np.ndarray] = None
    old_queries: List[str] = []
    old_gallery: List[str] = []
    old_best: Dict[str, Dict[str, str]] = {"gallery": {}, "queries": {}}

    if matrix_path.exists():
        try:
            with np.load(matrix_path) as data:
                matrix = np.asarray(data["verification"])
        except Exception as e:
            log.warning("Could not load existing verification matrix: %s", e)
            matrix = None

    if mapping_path.exists():
        try:
            with open(mapping_path, "r", encoding="utf-8") as f:
                m = json.load(f)
            old_queries = [str(x) for x in m.get("query_ids", [])]
            old_gallery = [str(x) for x in m.get("gallery_ids", [])]
        except Exception as e:
            log.warning("Could not load existing verification mapping: %s", e)

    if best_path.exists():
        try:
            with open(best_path, "r", encoding="utf-8") as f:
                b = json.load(f)
            if isinstance(b, dict):
                old_best = {
                    "gallery": {str(k): str(v) for k, v in (b.get("gallery", {}) or {}).items()},
                    "queries": {str(k): str(v) for k, v in (b.get("queries", {}) or {}).items()},
                }
        except Exception as e:
            log.warning("Could not load existing verification best_photos: %s", e)

    return matrix, old_queries, old_gallery, old_best


def run_verification_incremental_precompute(
    checkpoint_path: str,
    output_dir: Path,
    dirty_gallery_ids: Optional[List[str]] = None,
    dirty_query_ids: Optional[List[str]] = None,
    device: str = "cuda",
    batch_size: int = 16,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> Tuple[bool, str]:
    """
    Incrementally patch verification scores:
    - preserve existing unaffected pairs
    - recompute only missing/dirty query-gallery pairs
    """
    try:
        output_dir = Path(output_dir)

        if progress_callback:
            progress_callback("Selecting best photos...", 0, 100)

        gallery_best, query_best = select_best_photos()
        if not gallery_best:
            return False, "No gallery photos found (run image cache first)"
        if not query_best:
            return False, "No query photos found (run image cache first)"

        matrix_old, old_query_ids, old_gallery_ids, old_best = _load_existing_verification_artifacts(output_dir)

        query_ids = sorted(query_best.keys())
        gallery_ids = sorted(gallery_best.keys())
        q_index = {qid: i for i, qid in enumerate(query_ids)}
        g_index = {gid: i for i, gid in enumerate(gallery_ids)}

        n_q = len(query_ids)
        n_g = len(gallery_ids)
        total_pairs = n_q * n_g
        if total_pairs == 0:
            return False, "No verification pairs to compute"

        # Initialize with zeros, then copy overlapping old pairs.
        matrix_new = np.zeros((n_q, n_g), dtype=np.float32)
        if (
            matrix_old is not None
            and len(old_query_ids) == matrix_old.shape[0]
            and len(old_gallery_ids) == matrix_old.shape[1]
        ):
            old_q_index = {qid: i for i, qid in enumerate(old_query_ids)}
            old_g_index = {gid: i for i, gid in enumerate(old_gallery_ids)}
            overlap_q = [qid for qid in query_ids if qid in old_q_index]
            overlap_g = [gid for gid in gallery_ids if gid in old_g_index]
            for qid in overlap_q:
                qi_new = q_index[qid]
                qi_old = old_q_index[qid]
                for gid in overlap_g:
                    gi_new = g_index[gid]
                    gi_old = old_g_index[gid]
                    matrix_new[qi_new, gi_new] = float(matrix_old[qi_old, gi_old])

        # Dirty detection = explicit dirty IDs + changed best-photo path + new IDs.
        dirty_q = set(dirty_query_ids or [])
        dirty_g = set(dirty_gallery_ids or [])
        old_best_q = old_best.get("queries", {})
        old_best_g = old_best.get("gallery", {})

        for qid, p in query_best.items():
            if old_best_q.get(qid) != p:
                dirty_q.add(qid)
        for gid, p in gallery_best.items():
            if old_best_g.get(gid) != p:
                dirty_g.add(gid)
        for qid in query_ids:
            if qid not in old_query_ids:
                dirty_q.add(qid)
        for gid in gallery_ids:
            if gid not in old_gallery_ids:
                dirty_g.add(gid)

        # Determine which pairs must be recomputed.
        pairs_to_compute: List[Tuple[str, str]] = []
        pair_targets: List[Tuple[int, int]] = []
        for qid in query_ids:
            q_dirty = qid in dirty_q
            for gid in gallery_ids:
                if q_dirty or gid in dirty_g:
                    pairs_to_compute.append((query_best[qid], gallery_best[gid]))
                    pair_targets.append((q_index[qid], g_index[gid]))

        n_update = len(pairs_to_compute)
        if n_update > 0:
            precomputer = VerificationPrecomputer(
                checkpoint_path=checkpoint_path,
                device=device,
            )
            precomputer._ensure_inference()

            def batch_progress(current: int, total: int):
                if progress_callback:
                    progress_callback(
                        f"Verification incremental: {current}/{total} updated pairs",
                        current,
                        total,
                    )

            probs = precomputer._inference.predict_batch(
                pairs_to_compute,
                batch_size=batch_size,
                progress_callback=batch_progress,
            )
            for (qi, gi), score in zip(pair_targets, probs):
                matrix_new[qi, gi] = float(score)
        else:
            if progress_callback:
                progress_callback("Verification incremental: no dirty pairs", 1, 1)

        precomputer = VerificationPrecomputer(
            checkpoint_path=checkpoint_path,
            device=device,
        )
        precomputer.save_results(
            output_dir=output_dir,
            verification_matrix=matrix_new,
            query_ids=query_ids,
            gallery_ids=gallery_ids,
            gallery_best=gallery_best,
            query_best=query_best,
        )

        return (
            True,
            f"verification incremental ({n_update} updated / {total_pairs} total pairs)",
        )
    except Exception as e:
        log.error("Incremental verification precomputation failed: %s", e)
        import traceback
        traceback.print_exc()
        return False, f"Error: {str(e)}"


def estimate_verification_time(
    n_queries: int,
    n_gallery: int,
    is_gpu: bool = True,
) -> str:
    """
    Estimate verification computation time.
    
    Args:
        n_queries: Number of query identities
        n_gallery: Number of gallery identities
        is_gpu: Whether GPU is available
        
    Returns:
        Human-readable time estimate
    """
    total_pairs = n_queries * n_gallery
    
    # Rough estimates: GPU ~160 pairs/sec, CPU ~10 pairs/sec
    pairs_per_sec = 160 if is_gpu else 10
    seconds = total_pairs / pairs_per_sec
    
    if seconds < 60:
        return f"~{int(seconds)} seconds"
    elif seconds < 3600:
        return f"~{int(seconds / 60)} minutes"
    else:
        return f"~{seconds / 3600:.1f} hours"




