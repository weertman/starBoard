from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Dict, Optional
import os
from datetime import date
import re

# =============================================================================
# LEGACY COLUMN HEADERS (V1 - deprecated, kept for migration/reading old data)
# =============================================================================
GALLERY_HEADER_V1: List[str] = [
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

QUERIES_HEADER_V1: List[str] = [
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

# =============================================================================
# NEW COLUMN HEADERS (V2 - structured annotation schema)
# =============================================================================
GALLERY_HEADER_V2: List[str] = [
    "gallery_id",
    # Numeric measurements
    "num_apparent_arms",
    "num_total_arms",
    "tip_to_tip_size_cm",
    # Short arm coding
    "short_arm_code",
    # Stripe morphology
    "stripe_color",
    "stripe_order",
    "stripe_prominence",
    "stripe_extent",
    # Arm morphology
    "arm_color",
    "arm_thickness",
    # Central disc
    "central_disc_color",
    "papillae_central_disc_color",
    # Rosettes
    "rosette_color",
    "rosette_prominence",
    # Papillae stripes
    "papillae_stripe_color",
    # Madreporite
    "madreporite_color",
    # Reticulation
    "reticulation_order",
    # Overall
    "overall_color",
    # Notes
    "location",
    "unusual_observation",
    "health_observation",
]

QUERIES_HEADER_V2: List[str] = [
    "query_id",
    # Numeric measurements
    "num_apparent_arms",
    "num_total_arms",
    "tip_to_tip_size_cm",
    # Short arm coding
    "short_arm_code",
    # Stripe morphology
    "stripe_color",
    "stripe_order",
    "stripe_prominence",
    "stripe_extent",
    # Arm morphology
    "arm_color",
    "arm_thickness",
    # Central disc
    "central_disc_color",
    "papillae_central_disc_color",
    # Rosettes
    "rosette_color",
    "rosette_prominence",
    # Papillae stripes
    "papillae_stripe_color",
    # Madreporite
    "madreporite_color",
    # Reticulation
    "reticulation_order",
    # Overall
    "overall_color",
    # Notes
    "location",
    "unusual_observation",
    "health_observation",
]

# =============================================================================
# ACTIVE HEADERS (point to V2 for new data)
# =============================================================================
GALLERY_HEADER: List[str] = GALLERY_HEADER_V2
QUERIES_HEADER: List[str] = QUERIES_HEADER_V2

_MMDDYY = re.compile(r"^(\d{2})_(\d{2})_(\d{2})(?:_|$)")

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

def _parse_mmddyy(s: str) -> Optional[date]:
    m = _MMDDYY.match(s or "")
    if not m:
        return None
    mm, dd, yy = map(int, (m.group(1), m.group(2), m.group(3)))
    yy = 2000 + yy
    try:
        return date(yy, mm, dd)
    except Exception:
        return None

def last_observation_date(target: str, id_str: str) -> Optional[date]:
    """Return the most recent parsed MM_DD_YY date from encounter subfolders of <target>/<id_str>."""
    latest: Optional[date] = None
    for root in roots_for_read(target):
        base = root / id_str
        if not base.exists():
            continue
        for child in base.iterdir():
            if child.is_dir():
                d = _parse_mmddyy(child.name)
                if d and (latest is None or d > latest):
                    latest = d
    return latest

def last_observation_for_all(target: str) -> Dict[str, Optional[date]]:
    """
    Scan all IDs under all plausible roots for *target* ("Gallery" or "Queries")
    and return a mapping {id_str -> last_observation_date(...)}.
    """
    out: Dict[str, Optional[date]] = {}
    seen: set[str] = set()
    for root in roots_for_read(target):
        if not root.exists():
            continue
        for p in root.iterdir():
            if not p.is_dir():
                continue
            _id = p.name
            if _id in seen:
                continue
            seen.add(_id)
            out[_id] = last_observation_date(target, _id)
    return out