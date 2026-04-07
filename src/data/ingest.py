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
    id_str = (id_str or "").strip()
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


import re as _re

# Match encounter folder names like  6_10_2022_description  or  06_10_22
# Supports both M_D_YYYY (star_dataset) and MM_DD_YY (starBoard) conventions.
_ENC_DATE_RE = _re.compile(
    r"^(\d{1,2})_(\d{1,2})_(\d{2,4})"
)


def detect_folder_depth(root_dir) -> str:
    """Probe a directory to guess its structure depth.

    Returns one of:
        "single_id"  — root contains dated encounter folders directly
        "ids"        — root contains ID folders with dated encounters inside
        "grouped"    — root contains group folders containing ID folders
        "flat"       — root contains folders with images but no dated structure
        "empty"      — nothing recognisable found

    The heuristic checks the first few subdirectories at each level.
    """
    root = Path(root_dir)
    if not root.is_dir():
        return "empty"

    children = sorted(p for p in root.iterdir() if p.is_dir())
    if not children:
        return "empty"

    # Check if children themselves are dated encounter folders
    dated_children = sum(1 for c in children if _parse_encounter_date(c.name) is not None)
    if dated_children > 0 and dated_children >= len(children) * 0.5:
        return "single_id"

    # Check grandchildren: if most children contain dated sub-folders -> "ids" level
    sampled = children[:8]
    ids_with_encounters = 0
    ids_with_sub_ids = 0
    for child in sampled:
        grandchildren = sorted(p for p in child.iterdir() if p.is_dir())
        if not grandchildren:
            continue
        dated_gc = sum(1 for gc in grandchildren if _parse_encounter_date(gc.name) is not None)
        if dated_gc > 0 and dated_gc >= len(grandchildren) * 0.5:
            ids_with_encounters += 1
        else:
            # Check one more level: do grandchildren contain dated folders?
            for gc in grandchildren[:4]:
                ggc = [p for p in gc.iterdir() if p.is_dir()]
                dated_ggc = sum(1 for g in ggc if _parse_encounter_date(g.name) is not None)
                if dated_ggc > 0:
                    ids_with_sub_ids += 1
                    break

    if ids_with_encounters > 0 and ids_with_encounters >= len(sampled) * 0.3:
        return "ids"
    if ids_with_sub_ids > 0 and ids_with_sub_ids >= len(sampled) * 0.3:
        return "grouped"
    return "flat"


def _parse_encounter_date(folder_name: str) -> Optional[date]:
    """Parse a date from an encounter folder name.

    Accepts M_D_YYYY (e.g. 6_10_2022_notes), MM_DD_YY (e.g. 06_10_22),
    and similar variants.  Returns None if no date can be parsed.
    """
    m = _ENC_DATE_RE.match(folder_name)
    if not m:
        return None
    a, b, c = int(m.group(1)), int(m.group(2)), int(m.group(3))
    # Distinguish M_D_YYYY (year >= 100) from MM_DD_YY (year < 100)
    if c >= 100:
        month, day, year = a, b, c
    else:
        month, day, year = a, b, 2000 + c
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _encounter_suffix(folder_name: str) -> str:
    """Extract the descriptive suffix after the date portion of a folder name."""
    m = _ENC_DATE_RE.match(folder_name)
    if not m:
        return ""
    rest = folder_name[m.end():]
    return rest.lstrip("_")


# Return type for encounter-aware discovery: list of
#   (id_str, [(encounter_folder_name, parsed_date, [image_paths]), ...])
EncounterList = List[Tuple[str, date, List[Path]]]
IdWithEncounters = Tuple[str, EncounterList]


def discover_ids_with_encounters(
    parent_dir: Path,
) -> List[IdWithEncounters]:
    """Scan a directory of ID folders that each contain dated encounter sub-folders.

    Structure:  parent / individual_id / M_D_YYYY_description / images

    Encounter folders whose names cannot be parsed as dates are skipped.
    If an ID folder has images at its top level (no encounter sub-folders),
    those images are collected into a single encounter named after the ID
    folder with today's date.
    """
    parent = Path(parent_dir)
    out: List[IdWithEncounters] = []
    if not parent.exists() or not parent.is_dir():
        return out

    for id_dir in sorted(p for p in parent.iterdir() if p.is_dir()):
        id_str = id_dir.name
        encounters: EncounterList = []
        for enc_dir in sorted(p for p in id_dir.iterdir() if p.is_dir()):
            parsed = _parse_encounter_date(enc_dir.name)
            if parsed is None:
                log.debug("Skipping folder (no date match): %s", enc_dir)
                continue
            images = sorted(
                p for p in enc_dir.rglob("*")
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS
            )
            if images:
                encounters.append((enc_dir.name, parsed, images))
        if encounters:
            out.append((id_str, encounters))

    log.info("discover_ids_with_encounters: %d IDs under %s", len(out), parent)
    return out


def discover_grouped_ids_with_encounters(
    parent_dir: Path,
) -> List[Tuple[str, List[IdWithEncounters]]]:
    """Scan a directory with an extra grouping level above IDs.

    Structure:  parent / data_group / individual_id / M_D_YYYY_desc / images

    Returns list of (group_name, [IdWithEncounters, ...]).
    """
    parent = Path(parent_dir)
    out: List[Tuple[str, List[IdWithEncounters]]] = []
    if not parent.exists() or not parent.is_dir():
        return out

    for group_dir in sorted(p for p in parent.iterdir() if p.is_dir()):
        ids = discover_ids_with_encounters(group_dir)
        if ids:
            out.append((group_dir.name, ids))

    total_ids = sum(len(ids) for _, ids in out)
    log.info(
        "discover_grouped_ids_with_encounters: %d groups, %d IDs under %s",
        len(out), total_ids, parent,
    )
    return out