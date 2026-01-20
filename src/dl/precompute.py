"""
Precomputation pipeline for DL embeddings and similarity matrices.

Runs as a background worker (QThread) with progress signals.
Adapts settings based on available hardware (GPU vs CPU).
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Callable

import numpy as np

from PySide6.QtCore import QThread, Signal

from src.data import archive_paths as ap
from src.data.id_registry import list_ids
from src.data.image_index import list_image_files

from . import DL_AVAILABLE, DEVICE
from .registry import DLRegistry
from .reid_adapter import get_adapter
from .image_cache import CacheBuilder, get_cached_images, get_cache_status
from .outlier_detection import detect_outliers

log = logging.getLogger("starBoard.dl.precompute")


@dataclass
class HardwareProfile:
    """Hardware-adaptive settings for precomputation."""
    device: str
    batch_size: int
    use_tta: bool
    use_horizontal_flip: bool
    use_vertical_flip: bool
    use_mixed_precision: bool
    estimated_img_per_sec: float
    
    @classmethod
    def detect(cls) -> "HardwareProfile":
        """Auto-detect optimal settings for current hardware."""
        if DEVICE == "cuda":
            # GPU: full speed, all features
            return cls(
                device="cuda",
                batch_size=16,
                use_tta=True,
                use_horizontal_flip=True,
                use_vertical_flip=True,
                use_mixed_precision=True,
                estimated_img_per_sec=50.0
            )
        else:
            # CPU: optimize for speed
            return cls(
                device="cpu",
                batch_size=4,  # Smaller batches for memory
                use_tta=True,  # Only horizontal flip TTA
                use_horizontal_flip=True,
                use_vertical_flip=False,  # Skip vertical flip on CPU (2x speedup)
                use_mixed_precision=False,
                estimated_img_per_sec=3.0  # Conservative estimate
            )
    
    @classmethod
    def fast_cpu(cls) -> "HardwareProfile":
        """Fastest CPU settings (no TTA)."""
        return cls(
            device="cpu",
            batch_size=4,
            use_tta=False,
            use_horizontal_flip=False,
            use_vertical_flip=False,
            use_mixed_precision=False,
            estimated_img_per_sec=8.0
        )
    
    @classmethod
    def quality(cls) -> "HardwareProfile":
        """Maximum quality settings (full TTA)."""
        is_gpu = DEVICE == "cuda"
        return cls(
            device=DEVICE,
            batch_size=16 if is_gpu else 4,
            use_tta=True,
            use_horizontal_flip=True,
            use_vertical_flip=True,
            use_mixed_precision=is_gpu,
            estimated_img_per_sec=50.0 if is_gpu else 2.0
        )


def get_hardware_profile(mode: str = "auto") -> HardwareProfile:
    """
    Get hardware profile for precomputation.
    
    Args:
        mode: "auto" (detect), "fast" (speed priority), "quality" (accuracy priority)
    """
    if mode == "fast":
        if DEVICE == "cuda":
            return HardwareProfile.detect()  # GPU is already fast
        return HardwareProfile.fast_cpu()
    elif mode == "quality":
        return HardwareProfile.quality()
    else:
        return HardwareProfile.detect()

# Add star_identification to path for reranking
_STAR_ID_PATH = Path(__file__).parent.parent.parent / "star_identification"
if str(_STAR_ID_PATH) not in sys.path:
    sys.path.insert(0, str(_STAR_ID_PATH))


class PrecomputeWorker(QThread):
    """
    Background worker for precomputing embeddings and similarity.
    
    Adapts to hardware capabilities (GPU vs CPU) for optimal performance.
    
    Signals:
        progress(str, int, int): (message, current, total) progress updates
        finished(bool, str): (success, message) completion signal
    """
    
    progress = Signal(str, int, int)  # message, current, total
    finished = Signal(bool, str)       # success, message
    
    def __init__(self, 
                 model_key: str,
                 use_tta: bool = True,
                 use_reranking: bool = True,
                 batch_size: int = 8,
                 include_gallery: bool = True,
                 include_queries: bool = True,
                 only_pending: bool = False,
                 speed_mode: str = "auto",  # "auto", "fast", "quality"
                 include_verification: bool = True,
                 verification_model_key: Optional[str] = None,
                 parent=None):
        super().__init__(parent)
        
        self.model_key = model_key
        self.use_reranking = use_reranking
        self.include_gallery = include_gallery
        self.include_queries = include_queries
        self.only_pending = only_pending
        self.speed_mode = speed_mode
        
        # Verification settings
        self.include_verification = include_verification
        self.verification_model_key = verification_model_key
        
        # Get hardware-adaptive settings
        self.profile = get_hardware_profile(speed_mode)
        
        # Override with user preferences if explicitly set
        if not use_tta:
            self.profile.use_tta = False
            self.profile.use_horizontal_flip = False
            self.profile.use_vertical_flip = False
        
        if batch_size != 8:  # User explicitly set batch size
            self.profile.batch_size = batch_size
        
        self._cancelled = False
        self._start_time: Optional[float] = None
        self._images_processed = 0
        self._total_images_estimate = 0
    
    def cancel(self):
        """Request cancellation of the worker."""
        self._cancelled = True
    
    def _format_eta(self, seconds: float) -> str:
        """Format seconds as human-readable ETA."""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            return f"{int(seconds / 60)}m {int(seconds % 60)}s"
        else:
            hours = int(seconds / 3600)
            minutes = int((seconds % 3600) / 60)
            return f"{hours}h {minutes}m"
    
    def _emit_progress(self, message: str, current: int, total: int, images_done: int = 0):
        """Emit progress with ETA calculation."""
        if self._start_time is not None and images_done > 0:
            elapsed = time.time() - self._start_time
            img_per_sec = images_done / elapsed if elapsed > 0 else self.profile.estimated_img_per_sec
            
            remaining_images = self._total_images_estimate - images_done
            eta_seconds = remaining_images / img_per_sec if img_per_sec > 0 else 0
            
            eta_str = self._format_eta(eta_seconds)
            speed_str = f"{img_per_sec:.1f} img/s"
            message = f"{message} | {speed_str} | ETA: {eta_str}"
        
        self.progress.emit(message, current, total)
    
    def run(self):
        """Main worker thread."""
        try:
            self._run_precomputation()
        except Exception as e:
            log.error("Precomputation failed: %s", e)
            import traceback
            traceback.print_exc()
            self.finished.emit(False, f"Error: {str(e)}")
    
    def _run_precomputation(self):
        """
        Run the full precomputation pipeline with adaptive hardware settings.
        
        Two-phase approach:
        1. Build image cache (YOLO preprocessing + resize) - runs once
        2. Extract embeddings from cached images - much faster
        """
        if not DL_AVAILABLE:
            self.finished.emit(False, "PyTorch not available")
            return
        
        # Log hardware profile
        log.info("Hardware profile: device=%s, batch=%d, tta=%s (hflip=%s, vflip=%s)",
                 self.profile.device, self.profile.batch_size, self.profile.use_tta,
                 self.profile.use_horizontal_flip, self.profile.use_vertical_flip)
        
        # Load registry
        registry = DLRegistry.load()
        
        if self.model_key not in registry.models:
            self.finished.emit(False, f"Model not found: {self.model_key}")
            return
        
        model_entry = registry.models[self.model_key]
        adapter = get_adapter()
        
        # ============================================================
        # PHASE 1: Build image cache (YOLO preprocessing)
        # ============================================================
        self.progress.emit("Phase 1: Loading YOLO segmentation model...", 0, 100)
        
        yolo_preprocessor = None
        if adapter.load_yolo_preprocessor():
            yolo_preprocessor = adapter._yolo_preprocessor
            log.info("YOLO preprocessor loaded for image caching")
        else:
            log.warning("YOLO preprocessor not available - caching raw images")
        
        if self._cancelled:
            self.finished.emit(False, "Cancelled")
            return
        
        # Collect IDs to process
        gallery_ids = []
        query_ids = []
        
        if self.only_pending:
            gallery_ids = list(registry.pending_ids.gallery) if self.include_gallery else []
            query_ids = list(registry.pending_ids.queries) if self.include_queries else []
        else:
            if self.include_gallery:
                gallery_ids = list_ids("Gallery")
            if self.include_queries:
                query_ids = list_ids("Queries")
        
        total_ids = len(gallery_ids) + len(query_ids)
        if total_ids == 0:
            self.finished.emit(True, "No IDs to process")
            return
        
        # Build image cache
        cache_builder = CacheBuilder(yolo_preprocessor)
        
        def cache_progress(msg, cur, tot):
            self.progress.emit(f"Phase 1: {msg}", cur, tot * 2)  # Phase 1 is first half
        
        self._start_time = time.time()
        
        log.info("Phase 1: Building image cache for %d identities", total_ids)
        cached, failed, _ = cache_builder.build_full_cache(
            include_gallery=self.include_gallery,
            include_queries=self.include_queries,
            force=False,  # Don't rebuild existing cache
            progress_callback=cache_progress
        )
        
        cache_time = time.time() - self._start_time
        log.info("Phase 1 complete: %d cached, %d failed in %.1fs", cached, failed, cache_time)
        
        if self._cancelled:
            self.finished.emit(False, "Cancelled")
            return
        
        # ============================================================
        # PHASE 2: Extract embeddings from cached images
        # ============================================================
        self.progress.emit("Phase 2: Loading re-ID model...", total_ids, total_ids * 2)
        
        if not adapter.load_model(model_entry.checkpoint_path):
            self.finished.emit(False, "Failed to load model")
            return
        
        if self._cancelled:
            self.finished.emit(False, "Cancelled")
            return
        
        # Estimate total images for ETA calculation
        self._total_images_estimate = cached
        self._images_processed = 0
        self._start_time = time.time()
        
        # Create output directories
        model_dir = DLRegistry.get_model_data_dir(self.model_key)
        embeddings_dir = model_dir / "embeddings"
        similarity_dir = model_dir / "similarity"
        embeddings_dir.mkdir(parents=True, exist_ok=True)
        similarity_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract embeddings from CACHED images
        # Centroid embeddings (aggregated per identity)
        gallery_embeddings: Dict[str, np.ndarray] = {}
        query_embeddings: Dict[str, np.ndarray] = {}
        # Per-image embeddings (all images for each identity)
        gallery_image_embeddings: Dict[str, np.ndarray] = {}
        query_image_embeddings: Dict[str, np.ndarray] = {}
        # Image paths for each identity
        gallery_image_paths: Dict[str, List[str]] = {}
        query_image_paths: Dict[str, List[str]] = {}
        total_images = 0
        
        current = 0
        
        # Process gallery
        for gid in gallery_ids:
            if self._cancelled:
                self.finished.emit(False, "Cancelled")
                return
            
            current += 1
            
            # Use cached images instead of originals
            cached_images = get_cached_images("Gallery", gid)
            
            self._emit_progress(f"Phase 2 - Gallery: {gid} ({len(cached_images)} imgs)", 
                               total_ids + current, total_ids * 2, self._images_processed)
            
            if not cached_images:
                continue
            
            # Extract from cached images (skip YOLO since already preprocessed)
            embeddings = adapter.extract_batch(
                [str(p) for p in cached_images],
                use_tta=self.profile.use_tta,
                batch_size=self.profile.batch_size,
                use_horizontal_flip=self.profile.use_horizontal_flip,
                use_vertical_flip=self.profile.use_vertical_flip,
                use_yolo_preprocessing=False  # Already preprocessed!
            )
            
            if embeddings is not None and len(embeddings) > 0:
                image_paths = [str(p) for p in cached_images]
                
                # Store ALL per-image embeddings and paths
                gallery_image_embeddings[gid] = embeddings.copy()
                gallery_image_paths[gid] = image_paths
                
                # Aggregate with outlier rejection (z-score based) for centroid
                result = detect_outliers(
                    embeddings,
                    image_paths=image_paths,
                    id_str=gid
                )
                gallery_embeddings[gid] = result.inlier_embedding
                total_images += result.n_inliers  # Count only inliers
                self._images_processed += len(cached_images)
        
        # Process queries
        for qid in query_ids:
            if self._cancelled:
                self.finished.emit(False, "Cancelled")
                return
            
            current += 1
            
            # Use cached images
            cached_images = get_cached_images("Queries", qid)
            
            self._emit_progress(f"Phase 2 - Query: {qid} ({len(cached_images)} imgs)", 
                               total_ids + current, total_ids * 2, self._images_processed)
            
            if not cached_images:
                continue
            
            # Extract from cached images (skip YOLO)
            embeddings = adapter.extract_batch(
                [str(p) for p in cached_images],
                use_tta=self.profile.use_tta,
                batch_size=self.profile.batch_size,
                use_horizontal_flip=self.profile.use_horizontal_flip,
                use_vertical_flip=self.profile.use_vertical_flip,
                use_yolo_preprocessing=False  # Already preprocessed!
            )
            
            if embeddings is not None and len(embeddings) > 0:
                image_paths = [str(p) for p in cached_images]
                
                # Store ALL per-image embeddings and paths
                query_image_embeddings[qid] = embeddings.copy()
                query_image_paths[qid] = image_paths
                
                # Aggregate with outlier rejection (z-score based) for centroid
                result = detect_outliers(
                    embeddings,
                    image_paths=image_paths,
                    id_str=qid
                )
                query_embeddings[qid] = result.inlier_embedding
                total_images += result.n_inliers
                self._images_processed += len(cached_images)
        
        if self._cancelled:
            self.finished.emit(False, "Cancelled")
            return
        
        if not gallery_embeddings or not query_embeddings:
            self.finished.emit(False, "No embeddings extracted")
            return
        
        # Build matrices
        self.progress.emit("Computing similarity matrix...", total_ids, total_ids)
        
        gallery_ids_sorted = sorted(gallery_embeddings.keys())
        query_ids_sorted = sorted(query_embeddings.keys())
        
        gallery_matrix = np.stack([gallery_embeddings[gid] for gid in gallery_ids_sorted])
        query_matrix = np.stack([query_embeddings[qid] for qid in query_ids_sorted])
        
        # Compute similarity with optional re-ranking
        if self.use_reranking:
            try:
                from megastarid.inference import rerank
                
                # Re-ranking returns distance, convert to similarity
                distance = rerank(
                    query_embeddings=query_matrix,
                    gallery_embeddings=gallery_matrix,
                    k1=20,
                    k2=6,
                    lambda_value=0.3
                )
                similarity = 1.0 - distance
                
            except ImportError:
                log.warning("Could not import rerank, using cosine similarity")
                similarity = query_matrix @ gallery_matrix.T
        else:
            similarity = query_matrix @ gallery_matrix.T
        
        if self._cancelled:
            self.finished.emit(False, "Cancelled")
            return
        
        # Save results
        self.progress.emit("Saving centroid embeddings...", total_ids, total_ids)
        
        # Save centroid embeddings (aggregated per identity)
        np.savez_compressed(
            embeddings_dir / "gallery_embeddings.npz",
            **{gid: gallery_embeddings[gid] for gid in gallery_ids_sorted}
        )
        np.savez_compressed(
            embeddings_dir / "query_embeddings.npz",
            **{qid: query_embeddings[qid] for qid in query_ids_sorted}
        )
        
        # Save per-image embeddings
        self.progress.emit("Saving per-image embeddings...", total_ids, total_ids)
        np.savez_compressed(
            embeddings_dir / "gallery_image_embeddings.npz",
            **{gid: gallery_image_embeddings[gid] for gid in gallery_ids_sorted if gid in gallery_image_embeddings}
        )
        np.savez_compressed(
            embeddings_dir / "query_image_embeddings.npz",
            **{qid: query_image_embeddings[qid] for qid in query_ids_sorted if qid in query_image_embeddings}
        )
        
        # Save image paths mapping
        with open(embeddings_dir / "gallery_image_paths.json", 'w', encoding='utf-8') as f:
            json.dump({gid: gallery_image_paths[gid] for gid in gallery_ids_sorted if gid in gallery_image_paths}, f, indent=2)
        with open(embeddings_dir / "query_image_paths.json", 'w', encoding='utf-8') as f:
            json.dump({qid: query_image_paths[qid] for qid in query_ids_sorted if qid in query_image_paths}, f, indent=2)
        
        # Compute image-to-image similarity matrix
        self.progress.emit("Computing image-to-image similarity...", total_ids, total_ids)
        
        # Build flat arrays of all image embeddings with index mappings
        gallery_img_list = []  # [(id, local_idx, path), ...]
        gallery_img_embeddings_flat = []
        for gid in gallery_ids_sorted:
            if gid in gallery_image_embeddings:
                embs = gallery_image_embeddings[gid]
                paths = gallery_image_paths[gid]
                for local_idx, (emb, path) in enumerate(zip(embs, paths)):
                    gallery_img_list.append({"id": gid, "local_idx": local_idx, "path": path})
                    gallery_img_embeddings_flat.append(emb)
        
        query_img_list = []
        query_img_embeddings_flat = []
        for qid in query_ids_sorted:
            if qid in query_image_embeddings:
                embs = query_image_embeddings[qid]
                paths = query_image_paths[qid]
                for local_idx, (emb, path) in enumerate(zip(embs, paths)):
                    query_img_list.append({"id": qid, "local_idx": local_idx, "path": path})
                    query_img_embeddings_flat.append(emb)
        
        if gallery_img_embeddings_flat and query_img_embeddings_flat:
            gallery_img_matrix = np.stack(gallery_img_embeddings_flat)
            query_img_matrix = np.stack(query_img_embeddings_flat)
            
            # Compute image-to-image cosine similarity
            image_similarity = query_img_matrix @ gallery_img_matrix.T
            
            # Save image-level similarity
            np.savez_compressed(
                similarity_dir / "image_similarity_matrix.npz",
                similarity=image_similarity.astype(np.float32)
            )
            
            # Save image index mappings
            with open(similarity_dir / "gallery_image_index.json", 'w', encoding='utf-8') as f:
                json.dump(gallery_img_list, f, indent=2)
            with open(similarity_dir / "query_image_index.json", 'w', encoding='utf-8') as f:
                json.dump(query_img_list, f, indent=2)
            
            log.info("Image similarity matrix: %d query images x %d gallery images", 
                     len(query_img_list), len(gallery_img_list))
        
        # Save centroid similarity matrix
        np.savez_compressed(
            similarity_dir / "query_gallery_scores.npz",
            similarity=similarity.astype(np.float32)
        )
        
        # Save ID mapping
        with open(similarity_dir / "id_mapping.json", 'w', encoding='utf-8') as f:
            json.dump({
                "query_ids": query_ids_sorted,
                "gallery_ids": gallery_ids_sorted
            }, f, indent=2)
        
        # Save metadata
        with open(similarity_dir / "metadata.json", 'w', encoding='utf-8') as f:
            json.dump({
                "use_tta": self.profile.use_tta,
                "use_reranking": self.use_reranking,
                "embedding_dim": adapter.get_embedding_dim(),
                "image_size": adapter.get_image_size(),
                "n_gallery_images": len(gallery_img_list),
                "n_query_images": len(query_img_list)
            }, f, indent=2)
        
        # Update registry
        registry.mark_precomputed(
            self.model_key,
            gallery_count=len(gallery_embeddings),
            query_count=len(query_embeddings),
            image_count=total_images
        )
        
        # Clear lookup cache to force reload
        from .similarity_lookup import clear_cache
        clear_cache()
        
        # ============================================================
        # PHASE 4: Verification (if enabled)
        # ============================================================
        verification_msg = ""
        if self.include_verification:
            success, msg = self._run_verification_phase()
            if success:
                verification_msg = f", {msg}"
            elif msg:
                log.warning("Verification phase: %s", msg)
        
        self.finished.emit(
            True, 
            f"Completed: {len(gallery_embeddings)} gallery, {len(query_embeddings)} queries{verification_msg}"
        )
    
    def _run_verification_phase(self) -> tuple[bool, str]:
        """
        Run verification precomputation phase.
        
        Computes pairwise verification scores for best photos from each identity.
        
        Returns:
            (success, message) tuple
        """
        from .registry import DLRegistry, DEFAULT_VERIFICATION_KEY
        
        registry = DLRegistry.load()
        
        # Determine which verification model to use
        model_key = self.verification_model_key
        if model_key is None:
            # Use default if available
            if DEFAULT_VERIFICATION_KEY in registry.verification_models:
                model_key = DEFAULT_VERIFICATION_KEY
            elif registry.verification_models:
                model_key = next(iter(registry.verification_models.keys()))
            else:
                return False, "No verification model registered"
        
        if model_key not in registry.verification_models:
            return False, f"Verification model not found: {model_key}"
        
        verif_entry = registry.verification_models[model_key]
        checkpoint_path = verif_entry.checkpoint_path
        
        if not Path(checkpoint_path).exists():
            return False, f"Verification checkpoint not found: {checkpoint_path}"
        
        self.progress.emit("Phase 4: Loading verification model...", 0, 100)
        
        if self._cancelled:
            return False, "Cancelled"
        
        try:
            from .verification_precompute import (
                select_best_photos,
                VerificationPrecomputer,
            )
            
            # Select best photos
            self.progress.emit("Phase 4: Selecting best photos...", 5, 100)
            gallery_best, query_best = select_best_photos()
            
            if not gallery_best or not query_best:
                return False, "No best photos found (run embedding phase first)"
            
            n_pairs = len(gallery_best) * len(query_best)
            
            if self._cancelled:
                return False, "Cancelled"
            
            # Create precomputer
            precomputer = VerificationPrecomputer(
                checkpoint_path=checkpoint_path,
                device=self.profile.device,
            )
            
            # Progress callback for verification
            def verif_progress(msg: str, current: int, total: int):
                # Scale to 10-90 range for phase 4
                scaled = 10 + int(80 * current / max(total, 1))
                self.progress.emit(f"Phase 4: {msg}", scaled, 100)
            
            # Compute all pairs
            verification_matrix, query_ids, gallery_ids = precomputer.compute_all_pairs(
                gallery_best=gallery_best,
                query_best=query_best,
                batch_size=self.profile.batch_size,
                progress_callback=verif_progress,
            )
            
            if self._cancelled:
                return False, "Cancelled"
            
            # Save results
            self.progress.emit("Phase 4: Saving verification results...", 95, 100)
            
            output_dir = DLRegistry.get_verification_model_data_dir(model_key) / "verification"
            precomputer.save_results(
                output_dir=output_dir,
                verification_matrix=verification_matrix,
                query_ids=query_ids,
                gallery_ids=gallery_ids,
                gallery_best=gallery_best,
                query_best=query_best,
            )
            
            # Update registry
            registry.mark_verification_precomputed(model_key, n_pairs)
            
            # Clear verification lookup cache
            from .verification_lookup import clear_verification_cache
            clear_verification_cache()
            
            return True, f"verification ({n_pairs} pairs)"
            
        except Exception as e:
            log.error("Verification phase failed: %s", e)
            import traceback
            traceback.print_exc()
            return False, f"Verification error: {str(e)}"


def estimate_time(gallery_count: int, query_count: int, image_estimate: int = 10) -> str:
    """
    Estimate precomputation time.
    
    Args:
        gallery_count: Number of gallery IDs
        query_count: Number of query IDs
        image_estimate: Average images per ID
        
    Returns:
        Human-readable time estimate
    """
    total_images = (gallery_count + query_count) * image_estimate
    
    # Rough estimates: GPU ~50 img/s, CPU ~5 img/s
    if DEVICE == "cuda":
        seconds = total_images / 50
    else:
        seconds = total_images / 5
    
    # Add overhead for re-ranking
    seconds += (gallery_count * query_count) / 10000  # ~100k pairs/sec
    
    if seconds < 60:
        return f"~{int(seconds)} seconds"
    elif seconds < 3600:
        return f"~{int(seconds / 60)} minutes"
    else:
        return f"~{seconds / 3600:.1f} hours"


def run_precomputation_sync(
    model_key: str,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
    use_tta: bool = True,
    use_reranking: bool = True,
    batch_size: int = 8
) -> tuple[bool, str]:
    """
    Run precomputation synchronously (blocking).
    
    For use in scripts or when a blocking call is acceptable.
    
    Args:
        model_key: The model key to precompute for
        progress_callback: Optional callback(message, current, total)
        use_tta: Whether to use test-time augmentation
        use_reranking: Whether to use k-reciprocal re-ranking
        batch_size: Batch size for embedding extraction
        
    Returns:
        (success, message) tuple
    """
    if not DL_AVAILABLE:
        return False, "PyTorch not available"
    
    # Load registry
    registry = DLRegistry.load()
    
    if model_key not in registry.models:
        return False, f"Model not found: {model_key}"
    
    model_entry = registry.models[model_key]
    
    # Load model
    if progress_callback:
        progress_callback("Loading model...", 0, 100)
    
    adapter = get_adapter()
    if not adapter.load_model(model_entry.checkpoint_path):
        return False, "Failed to load model"
    
    # Collect IDs
    gallery_ids = list_ids("Gallery")
    query_ids = list_ids("Queries")
    
    total_ids = len(gallery_ids) + len(query_ids)
    if total_ids == 0:
        return True, "No IDs to process"
    
    # Create output directories
    model_dir = DLRegistry.get_model_data_dir(model_key)
    embeddings_dir = model_dir / "embeddings"
    similarity_dir = model_dir / "similarity"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    similarity_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract embeddings
    gallery_embeddings: Dict[str, np.ndarray] = {}
    query_embeddings: Dict[str, np.ndarray] = {}
    total_images = 0
    current = 0
    
    for gid in gallery_ids:
        current += 1
        if progress_callback:
            progress_callback(f"Gallery: {gid}", current, total_ids)
        
        images = list_image_files("Gallery", gid)
        if not images:
            continue
        
        embeddings = adapter.extract_batch(
            [str(p) for p in images],
            use_tta=use_tta,
            batch_size=batch_size
        )
        
        if embeddings is not None and len(embeddings) > 0:
            agg = embeddings.mean(axis=0)
            agg = agg / (np.linalg.norm(agg) + 1e-8)
            gallery_embeddings[gid] = agg
            total_images += len(images)
    
    for qid in query_ids:
        current += 1
        if progress_callback:
            progress_callback(f"Query: {qid}", current, total_ids)
        
        images = list_image_files("Queries", qid)
        if not images:
            continue
        
        embeddings = adapter.extract_batch(
            [str(p) for p in images],
            use_tta=use_tta,
            batch_size=batch_size
        )
        
        if embeddings is not None and len(embeddings) > 0:
            agg = embeddings.mean(axis=0)
            agg = agg / (np.linalg.norm(agg) + 1e-8)
            query_embeddings[qid] = agg
            total_images += len(images)
    
    if not gallery_embeddings or not query_embeddings:
        return False, "No embeddings extracted"
    
    # Compute similarity
    if progress_callback:
        progress_callback("Computing similarity...", total_ids, total_ids)
    
    gallery_ids_sorted = sorted(gallery_embeddings.keys())
    query_ids_sorted = sorted(query_embeddings.keys())
    
    gallery_matrix = np.stack([gallery_embeddings[gid] for gid in gallery_ids_sorted])
    query_matrix = np.stack([query_embeddings[qid] for qid in query_ids_sorted])
    
    if use_reranking:
        try:
            from megastarid.inference import rerank
            distance = rerank(query_matrix, gallery_matrix, k1=20, k2=6, lambda_value=0.3)
            similarity = 1.0 - distance
        except ImportError:
            similarity = query_matrix @ gallery_matrix.T
    else:
        similarity = query_matrix @ gallery_matrix.T
    
    # Save
    np.savez_compressed(
        embeddings_dir / "gallery_embeddings.npz",
        **{gid: gallery_embeddings[gid] for gid in gallery_ids_sorted}
    )
    np.savez_compressed(
        embeddings_dir / "query_embeddings.npz",
        **{qid: query_embeddings[qid] for qid in query_ids_sorted}
    )
    np.savez_compressed(
        similarity_dir / "query_gallery_scores.npz",
        similarity=similarity.astype(np.float32)
    )
    
    with open(similarity_dir / "id_mapping.json", 'w', encoding='utf-8') as f:
        json.dump({
            "query_ids": query_ids_sorted,
            "gallery_ids": gallery_ids_sorted
        }, f, indent=2)
    
    # Update registry
    registry.mark_precomputed(
        model_key,
        gallery_count=len(gallery_embeddings),
        query_count=len(query_embeddings),
        image_count=total_images
    )
    
    from .similarity_lookup import clear_cache
    clear_cache()
    
    return True, f"Completed: {len(gallery_embeddings)} gallery, {len(query_embeddings)} queries"


def rerun_embeddings(
    model_key: str = None,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
    use_tta: bool = True,
    use_reranking: bool = True,
    batch_size: int = 8
) -> tuple[bool, str]:
    """
    Reset and rerun embedding extraction + similarity computation.
    
    This is a convenience function that:
    1. Resets the precomputed state (deletes embeddings/similarity files)
    2. Keeps the YOLO image cache intact
    3. Reruns the full embedding pipeline
    
    Use this when you've changed outlier detection, aggregation, or other
    embedding-related logic and want to recompute everything.
    
    Args:
        model_key: Model to rerun (uses active or default if None)
        progress_callback: Optional callback(message, current, total)
        use_tta: Whether to use test-time augmentation
        use_reranking: Whether to use k-reciprocal re-ranking
        batch_size: Batch size for embedding extraction
        
    Returns:
        (success, message) tuple
    """
    if not DL_AVAILABLE:
        return False, "PyTorch not available"
    
    # Get model key
    registry = DLRegistry.load()
    
    if model_key is None:
        model_key = registry.active_model
        if model_key is None:
            # Use first available model
            if registry.models:
                model_key = next(iter(registry.models.keys()))
            else:
                return False, "No models registered"
    
    if model_key not in registry.models:
        return False, f"Model not found: {model_key}"
    
    # Reset (delete embeddings, keep image cache)
    if progress_callback:
        progress_callback("Resetting precomputed state...", 0, 100)
    
    registry.reset_precomputed(model_key, delete_files=True)
    
    # Rerun
    return run_precomputation_sync(
        model_key=model_key,
        progress_callback=progress_callback,
        use_tta=use_tta,
        use_reranking=use_reranking,
        batch_size=batch_size
    )

