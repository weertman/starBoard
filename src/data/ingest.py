from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple
import shutil
import logging

from .validators import validate_mmddyy_string

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

def place_images(target_root: Path, id_str: str, encounter_dir_name: str, files: Sequence[Path], move: bool = False) -> IngestReport:
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
    return report

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