# src/data/image_index.py
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple
import logging
from src.data.archive_paths import roots_for_read

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
log = logging.getLogger("starBoard.data.images")


@lru_cache(maxsize=512)
def list_image_files(target: str, id_str: str) -> Tuple[Path, ...]:
    """
    Return images under all plausible roots for target/id, in deterministic order.
    
    Returns a tuple (for caching hashability); callers can iterate or convert to list.
    """
    files: List[Path] = []
    for root in roots_for_read(target):
        base = root / id_str
        if not base.exists():
            continue
        # Collect images under encounters/* (sorted by path)
        for p in sorted(base.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                files.append(p)
    log.info("image_list target=%s id=%s count=%d", target, id_str, len(files))
    return tuple(files)


def invalidate_image_cache(target: str = None, id_str: str = None) -> None:
    """
    Clear the list_image_files cache.
    
    Call after adding/removing images. If target and id_str are provided,
    only that specific entry is invalidated; otherwise the entire cache is cleared.
    """
    if target is not None and id_str is not None:
        # Invalidate a specific cache entry
        try:
            # lru_cache doesn't support single key invalidation, so clear all
            list_image_files.cache_clear()
            log.debug("Invalidated image cache for %s/%s (full clear)", target, id_str)
        except Exception:
            list_image_files.cache_clear()
    else:
        list_image_files.cache_clear()
        log.debug("Invalidated entire image cache")
