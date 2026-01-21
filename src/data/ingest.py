from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import List, Optional, Sequence, Tuple
import shutil
import logging

from .validators import validate_mmddyy_string
from .id_registry import invalidate_id_cache
from .image_index import invalidate_image_cache

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
log = logging.getLogger("starBoard.data.ingest")

@dataclass
class FileOp:
    src: Path
    dest: Path
    action: str  # "copied" | "moved"
    renamed: bool = False

@dataclass
class IngestReport:
    target_root: Path
    id_str: str
    encounter_name: str
    ops: List[FileOp]
    errors: List[str]

def ensure_encounter_name(year: int, month: int, day: int, suffix: str = "") -> str:
    yy = year % 100
    base = f"{month:02d}_{day:02d}_{yy:02d}"
    if suffix:
        return f"{base}_{suffix}"
    return base

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _ensure_unique_path(dest: Path) -> Path:
    if not dest.exists():
        return dest
    stem = dest.stem
    suffix = dest.suffix
    parent = dest.parent
    n = 1
    while True:
        candidate = parent / f"{stem} ({n}){suffix}"
        if not candidate.exists():
            return candidate
        n += 1

def place_images(
    target_root: Path,
    id_str: str,
    encounter_dir_name: str,
    files: Sequence[Path],
    move: bool = False,
    observation_date: Optional[date] = None,
) -> IngestReport:
    """
    Place images into the archive structure.
    
    Args:
        target_root: Root directory (gallery or queries root)
        id_str: The ID (gallery_id or query_id)
        encounter_dir_name: Name of the encounter folder (MM_DD_YY format)
        files: List of image files to place
        move: If True, move files instead of copying
        observation_date: The observation date to save to encounter_dates.csv.
                         If None, will be parsed from encounter_dir_name.
    
    Returns:
        IngestReport with operation details and any errors
    """
    report = IngestReport(target_root=target_root, id_str=id_str, encounter_name=encounter_dir_name, ops=[], errors=[])

    v = validate_mmddyy_string(encounter_dir_name)
    if not v.ok:
        report.errors.append(v.message)
        return report

    dest_dir = target_root / id_str / encounter_dir_name
    _ensure_dir(dest_dir)

    for f in files:
        f = Path(f)
        if not f.exists() or not f.is_file():
            report.errors.append(f"Missing file: {f}")
            continue
        if f.suffix.lower() not in IMAGE_EXTS:
            continue

        dest_path = dest_dir / f.name
        final_dest = _ensure_unique_path(dest_path)
        renamed = (final_dest.name != dest_path.name)

        try:
            if move:
                shutil.move(str(f), str(final_dest))
                action = "moved"
            else:
                shutil.copy2(str(f), str(final_dest))
                action = "copied"
            report.ops.append(FileOp(src=f, dest=final_dest, action=action, renamed=renamed))
        except Exception as e:
            report.errors.append(f"Failed to transfer {f.name}: {e}")

    log.info("Ingested %d files into %s/%s", len(report.ops), id_str, encounter_dir_name)
    if report.errors:
        log.warning("Ingest errors: %s", "; ".join(report.errors))
    
    # Track new ID as pending for DL precomputation (best-effort)
    if report.ops:
        _track_pending_id(target_root, id_str)
        # Invalidate caches since new files/IDs may have been added
        invalidate_id_cache()
        invalidate_image_cache()
        
        # Save observation date to encounter_dates.csv
        _save_observation_date(target_root, id_str, encounter_dir_name, observation_date)
    
    return report


def _save_observation_date(
    target_root: Path,
    id_str: str,
    encounter_dir_name: str,
    observation_date: Optional[date],
) -> None:
    """
    Save the observation date to encounter_dates.csv.
    
    If observation_date is None, parses it from the encounter folder name.
    """
    try:
        from .encounter_info import set_encounter_date, _parse_mmddyy, invalidate_encounter_dates_cache
        
        # Determine target from root path
        root_name = target_root.name.lower()
        if "gallery" in root_name:
            target = "Gallery"
        elif "quer" in root_name:
            target = "Queries"
        else:
            log.debug("Unknown target for observation date: %s", target_root)
            return
        
        # Use provided date or parse from folder name
        if observation_date is None:
            observation_date = _parse_mmddyy(encounter_dir_name)
        
        if observation_date is not None:
            set_encounter_date(target, id_str, encounter_dir_name, observation_date)
            log.debug("Saved observation date %s for %s/%s/%s", 
                     observation_date.isoformat(), target, id_str, encounter_dir_name)
    except Exception as e:
        log.debug("Failed to save observation date: %s", e)


def _track_pending_id(target_root: Path, id_str: str):
    """Track an ID as pending for DL precomputation."""
    try:
        from src.dl.registry import DLRegistry
        registry = DLRegistry.load()
        
        # Determine target from root path
        root_name = target_root.name.lower()
        if "gallery" in root_name:
            target = "Gallery"
        elif "quer" in root_name:
            target = "Queries"
        else:
            return  # Unknown target
        
        registry.add_pending_id(target, id_str)
        log.debug("Tracked pending ID for DL: %s/%s", target, id_str)
    except ImportError:
        pass  # DL module not available
    except Exception as e:
        log.debug("Failed to track pending ID: %s", e)

def discover_ids_and_images(parent_dir: Path) -> List[Tuple[str, List[Path]]]:
    parent = Path(parent_dir)
    out: List[Tuple[str, List[Path]]] = []
    if not parent.exists() or not parent.is_dir():
        return out

    for child in sorted([p for p in parent.iterdir() if p.is_dir()]):
        id_str = child.name
        files: List[Path] = []
        for p in child.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                files.append(p)
        out.append((id_str, files))
    log.info("Discovered %d IDs under %s", len(out), str(parent))
    return out