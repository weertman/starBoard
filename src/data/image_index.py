# src/data/image_index.py
from __future__ import annotations
from pathlib import Path
from typing import List
import logging
from src.data.archive_paths import roots_for_read

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
log = logging.getLogger("starBoard.data.images")

def list_image_files(target: str, id_str: str) -> List[Path]:
    """Return images under all plausible roots for target/id, in deterministic order."""
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
    return files
