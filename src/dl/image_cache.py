"""
Image cache for YOLO-preprocessed images.

Creates a mirror dataset of cropped/resized images to speed up embedding extraction.
Structure mirrors archive/gallery and archive/queries but with preprocessed images.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image

from src.data import archive_paths as ap
from src.data.id_registry import list_ids
from src.data.image_index import list_image_files

log = logging.getLogger("starBoard.dl.image_cache")

# Cache location inside star_identification
CACHE_ROOT = Path(__file__).parent.parent.parent / "star_identification" / "precompute_cache"
CACHE_SIZE = 640  # Target size matching star_dataset_resized


def get_cache_root() -> Path:
    """Get the root directory for cached images."""
    return CACHE_ROOT


def get_cached_image_path(target: str, id_str: str, original_path: Path) -> Path:
    """
    Get the cached image path for an original image.
    
    Args:
        target: "Gallery" or "Queries"
        id_str: The identity ID
        original_path: Path to the original image
        
    Returns:
        Path where the cached image should be stored
    """
    cache_dir = CACHE_ROOT / target.lower() / id_str
    # Deterministic source token keeps duplicate stems (across encounters) distinct.
    try:
        source_key = str(original_path.resolve())
    except Exception:
        source_key = str(original_path)
    token = hashlib.sha1(source_key.encode("utf-8")).hexdigest()[:12]
    # Keep original stem so similarity lookup can still map by filename stem.
    cached_name = original_path.stem + ".png"
    return cache_dir / token / cached_name


def is_cached(target: str, id_str: str, original_path: Path) -> bool:
    """Check if an image is already cached."""
    cached_path = get_cached_image_path(target, id_str, original_path)
    return cached_path.exists()


def get_cached_images(target: str, id_str: str) -> List[Path]:
    """
    Get list of cached images for an identity.
    
    Args:
        target: "Gallery" or "Queries"
        id_str: The identity ID
        
    Returns:
        List of paths to cached images
    """
    cache_dir = CACHE_ROOT / target.lower() / id_str
    if not cache_dir.exists():
        return []
    
    return sorted(p for p in cache_dir.rglob("*.png") if p.is_file())


def sync_identity_cache(
    target: str,
    id_str: str,
    *,
    remove_stale: bool = True,
) -> Tuple[int, int, int]:
    """
    Sync one identity cache against current archive files.

    Returns:
        (live_count, cached_count_after_sync, stale_removed_count)
    """
    original_images = list_image_files(target, id_str)
    expected = {get_cached_image_path(target, id_str, p) for p in original_images}
    live_count = len(expected)

    cache_dir = CACHE_ROOT / target.lower() / id_str
    if not cache_dir.exists():
        return live_count, 0, 0

    cached_paths = [p for p in cache_dir.rglob("*.png") if p.is_file()]
    removed = 0

    if remove_stale:
        for p in cached_paths:
            if p not in expected:
                try:
                    p.unlink()
                    removed += 1
                except Exception as e:
                    log.warning("Failed removing stale cache file %s: %s", p, e)

        # Remove empty hashed subfolders after pruning stale files.
        for d in sorted([p for p in cache_dir.rglob("*") if p.is_dir()], key=lambda p: len(p.parts), reverse=True):
            try:
                d.rmdir()
            except OSError:
                pass
            except Exception as e:
                log.debug("Failed pruning empty cache dir %s: %s", d, e)

    cached_after = len([p for p in cache_dir.rglob("*.png") if p.is_file()])
    return live_count, cached_after, removed


def get_cache_status() -> Dict[str, any]:
    """
    Get cache status information.
    
    Returns:
        Dict with cache statistics
    """
    if not CACHE_ROOT.exists():
        return {
            "exists": False,
            "gallery_ids": 0,
            "query_ids": 0,
            "total_images": 0,
            "size_mb": 0
        }
    
    gallery_dir = CACHE_ROOT / "gallery"
    queries_dir = CACHE_ROOT / "queries"
    
    gallery_ids = len(list(gallery_dir.iterdir())) if gallery_dir.exists() else 0
    query_ids = len(list(queries_dir.iterdir())) if queries_dir.exists() else 0
    
    total_images = 0
    total_size = 0
    
    for img_path in CACHE_ROOT.rglob("*.png"):
        total_images += 1
        total_size += img_path.stat().st_size
    
    return {
        "exists": True,
        "gallery_ids": gallery_ids,
        "query_ids": query_ids,
        "total_images": total_images,
        "size_mb": total_size / (1024 * 1024)
    }


def clear_cache():
    """Clear all cached images."""
    if CACHE_ROOT.exists():
        shutil.rmtree(CACHE_ROOT)
        log.info("Cleared image cache at %s", CACHE_ROOT)


def _resize_image(img: Image.Image, target_size: int = CACHE_SIZE) -> Image.Image:
    """
    Resize image to target size (preserving aspect ratio, fitting in square).
    
    Uses the same resizing approach as star_dataset_resized.
    """
    # Get current size
    width, height = img.size
    
    # Calculate scale to fit in target_size x target_size
    scale = target_size / max(width, height)
    
    if scale < 1.0:
        # Only downscale, never upscale
        new_width = int(width * scale)
        new_height = int(height * scale)
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    return img


def preprocess_and_cache_image(
    target: str,
    id_str: str, 
    original_path: Path,
    yolo_preprocessor,
    force: bool = False
) -> Optional[Path]:
    """
    Preprocess an image with YOLO and cache the result.
    
    Args:
        target: "Gallery" or "Queries"
        id_str: The identity ID
        original_path: Path to the original image
        yolo_preprocessor: YOLOPreprocessor instance (or None to skip YOLO)
        force: If True, overwrite existing cached image
        
    Returns:
        Path to the cached image, or None if failed
    """
    cached_path = get_cached_image_path(target, id_str, original_path)
    
    # Check if already cached
    if cached_path.exists() and not force:
        return cached_path
    
    try:
        # Create output directory
        cached_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Process with YOLO if available
        if yolo_preprocessor is not None:
            processed_img = yolo_preprocessor.process_image(str(original_path))
            if processed_img is None:
                log.warning("YOLO detection failed for %s, using original", original_path.name)
                processed_img = Image.open(original_path).convert('RGB')
        else:
            # No YOLO, just load original
            processed_img = Image.open(original_path).convert('RGB')
        
        # Resize to cache size
        resized_img = _resize_image(processed_img, CACHE_SIZE)
        
        # Save as PNG
        resized_img.save(cached_path, "PNG", optimize=True)
        
        return cached_path
        
    except Exception as e:
        log.error("Failed to cache image %s: %s", original_path, e)
        return None


def build_cache_for_identity(
    target: str,
    id_str: str,
    yolo_preprocessor,
    force: bool = False
) -> Tuple[int, int]:
    """
    Build cache for all images of an identity.
    
    Args:
        target: "Gallery" or "Queries"
        id_str: The identity ID
        yolo_preprocessor: YOLOPreprocessor instance
        force: If True, rebuild existing cache
        
    Returns:
        (cached_count, failed_count)
    """
    original_images = list_image_files(target, id_str)
    
    cached = 0
    failed = 0
    
    for img_path in original_images:
        result = preprocess_and_cache_image(
            target, id_str, img_path, yolo_preprocessor, force
        )
        if result is not None:
            cached += 1
        else:
            failed += 1
    
    return cached, failed


class CacheBuilder:
    """
    Builds the image cache with progress reporting.
    
    Use this from the precomputation pipeline to ensure
    all images are cached before embedding extraction.
    """
    
    def __init__(self, yolo_preprocessor=None):
        self.yolo_preprocessor = yolo_preprocessor
        self._cancelled = False
    
    def cancel(self):
        """Request cancellation."""
        self._cancelled = True
    
    def build_full_cache(
        self,
        include_gallery: bool = True,
        include_queries: bool = True,
        force: bool = False,
        progress_callback=None
    ) -> Tuple[int, int, int]:
        """
        Build cache for all identities.
        
        Args:
            include_gallery: Include gallery IDs
            include_queries: Include query IDs
            force: Rebuild existing cache
            progress_callback: Callable(message, current, total)
            
        Returns:
            (total_cached, total_failed, total_stale_removed)
        """
        gallery_ids = list_ids("Gallery") if include_gallery else []
        query_ids = list_ids("Queries") if include_queries else []
        
        total_ids = len(gallery_ids) + len(query_ids)
        current = 0
        
        total_cached = 0
        total_failed = 0
        total_stale_removed = 0
        
        # Process gallery
        for gid in gallery_ids:
            if self._cancelled:
                return total_cached, total_failed, total_stale_removed
            
            current += 1
            if progress_callback:
                progress_callback(f"Caching Gallery: {gid}", current, total_ids)

            live_n, cached_n, stale_removed = sync_identity_cache("Gallery", gid)
            if stale_removed > 0:
                total_stale_removed += stale_removed
                log.info(
                    "Pruned stale cache files target=Gallery id=%s removed=%d live=%d cached_after=%d",
                    gid, stale_removed, live_n, cached_n
                )
            
            cached, failed = build_cache_for_identity(
                "Gallery", gid, self.yolo_preprocessor, force
            )
            total_cached += cached
            total_failed += failed
        
        # Process queries
        for qid in query_ids:
            if self._cancelled:
                return total_cached, total_failed, total_stale_removed
            
            current += 1
            if progress_callback:
                progress_callback(f"Caching Queries: {qid}", current, total_ids)

            live_n, cached_n, stale_removed = sync_identity_cache("Queries", qid)
            if stale_removed > 0:
                total_stale_removed += stale_removed
                log.info(
                    "Pruned stale cache files target=Queries id=%s removed=%d live=%d cached_after=%d",
                    qid, stale_removed, live_n, cached_n
                )
            
            cached, failed = build_cache_for_identity(
                "Queries", qid, self.yolo_preprocessor, force
            )
            total_cached += cached
            total_failed += failed
        
        log.info(
            "Cache build complete: %d cached, %d failed, %d stale_removed",
            total_cached, total_failed, total_stale_removed
        )
        
        return total_cached, total_failed, total_stale_removed
    
    def ensure_cached(self, target: str, id_str: str) -> List[Path]:
        """
        Ensure images for an identity are cached, return cached paths.
        
        This is called during embedding extraction to get the cached paths.
        If not cached, processes and caches on the fly.
        
        Args:
            target: "Gallery" or "Queries"
            id_str: The identity ID
            
        Returns:
            List of paths to cached images
        """
        original_images = list_image_files(target, id_str)
        cached_paths = []
        
        for img_path in original_images:
            cached_path = get_cached_image_path(target, id_str, img_path)
            
            if cached_path.exists():
                cached_paths.append(cached_path)
            else:
                # Cache on the fly
                result = preprocess_and_cache_image(
                    target, id_str, img_path, self.yolo_preprocessor
                )
                if result is not None:
                    cached_paths.append(result)
        
        return cached_paths










