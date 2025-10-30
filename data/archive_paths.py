from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
import os

# Column headers (enforced order)
GALLERY_HEADER: List[str] = [
    "gallery_id",
    "Last location",
    "sex",
    "diameter_cm",
    "volume_ml",
    "num_apparent_arms",
    "num_arms",
    "short_arm_codes",
    "stripe_descriptions",
    "reticulation_descriptions",
    "rosette_descriptions",
    "madreporite_descriptions",
    "disk color",
    "arm color",
    "Other_descriptions",
]

QUERIES_HEADER: List[str] = [
    "query_id",
    "Last location",
    "sex",
    "diameter_cm",
    "volume_ml",
    "num_apparent_arms",
    "num_arms",
    "short_arm_codes",
    "stripe_descriptions",
    "reticulation_descriptions",
    "rosette_descriptions",
    "madreporite_descriptions",
    "disk color",
    "arm color",
    "Other_descriptions",
]


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def archive_root() -> Path:
    env = os.getenv("STARBOARD_ARCHIVE_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return project_root() / "archive"


def gallery_root() -> Path:
    return archive_root() / "gallery"


def queries_root(prefer_new: bool = True) -> Path:
    root = archive_root()
    q_new = root / "queries"
    q_legacy = root / "querries"
    if prefer_new and q_new.exists():
        return q_new
    if q_new.exists():
        return q_new
    return q_legacy


def has_legacy_querries() -> bool:
    return (archive_root() / "querries").exists()


def metadata_csv_for(target: str) -> Tuple[Path, List[str]]:
    """Preferred CSV path for WRITES (Gallery or Queries)."""
    t = target.lower()
    if t not in ("gallery", "queries"):
        raise ValueError(f"Unknown target '{target}', expected 'Gallery' or 'Queries'.")
    if t == "gallery":
        return gallery_root() / "gallery_metadata.csv", GALLERY_HEADER
    root = queries_root(prefer_new=True)
    fname = "queries_metadata.csv" if root.name == "queries" else "querries_metadata.csv"
    return root / fname, QUERIES_HEADER


def metadata_csv_paths_for_read(target: str) -> List[Path]:
    """Return ALL plausible CSVs to READ (handles queries + querries)."""
    if target.lower() == "gallery":
        return [gallery_root() / "gallery_metadata.csv"]
    root = archive_root()
    candidates = [
        root / "querries" / "querries_metadata.csv",  # legacy first
        root / "queries" / "queries_metadata.csv",    # new second (wins on collisions)
    ]
    return [p for p in candidates if p.exists()]


def roots_for_read(target: str) -> List[Path]:
    """Return all plausible archive roots to READ IDs from."""
    if target.lower() == "gallery":
        return [gallery_root()]
    root = archive_root()
    out: List[Path] = []
    if (root / "querries").exists():
        out.append(root / "querries")
    if (root / "queries").exists():
        out.append(root / "queries")
    if not out:
        out.append(queries_root(prefer_new=True))
    return out


def root_for(target: str) -> Path:
    """Preferred root for WRITES."""
    return gallery_root() if target.lower() == "gallery" else queries_root(prefer_new=True)


def id_column_name(target: str) -> str:
    return "gallery_id" if target.lower() == "gallery" else "query_id"
