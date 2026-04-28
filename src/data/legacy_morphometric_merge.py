from __future__ import annotations

"""
One-off importer for deprecated standalone morphometric measurements.

This module is intentionally narrow in scope. It is designed for the
`tmp_10-27-2025-samish-photo-sample-merge_to_gallery` dataset and imports the
selected legacy `mFolder_*` bundles into the current starBoard gallery-backed
measurement/archive model.

Default behavior is a dry run:

    python -m src.data.legacy_morphometric_merge

To execute the import:

    python -m src.data.legacy_morphometric_merge --apply

To backfill archive images for already-imported legacy measurements:

    python -m src.data.legacy_morphometric_merge --backfill-archive-images --apply
"""

import argparse
import filecmp
import json
import logging
import re
import shutil
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from src.data import archive_paths as ap
from src.data.csv_io import append_row, normalize_id_value, read_rows, last_row_per_id
from src.data.image_index import list_image_files
from src.data.ingest import ensure_encounter_name, place_images
from src.data.metadata_history import record_morphometric_import
from src.morphometric import get_measurements_root
from src.morphometric.data_bridge import (
    extract_starboard_fields,
    list_mfolders,
    load_morphometrics_from_mfolder,
)

log = logging.getLogger("starBoard.data.legacy_morphometric_merge")

LEGACY_SOURCE_ROOT_NAME = "tmp_10-27-2025-samish-photo-sample-merge_to_gallery"
LEGACY_REQUIRED_FILENAMES: Tuple[str, ...] = (
    "corrected_detection.json",
    "corrected_mask.png",
    "corrected_object.png",
    "morphometrics.json",
)
LEGACY_PROVENANCE_KEY = "legacy_source_mfolder"
ARCHIVE_IMAGE_EXTS = {".jpg", ".jpeg", ".jpe", ".jfif", ".png", ".tif", ".tiff", ".bmp", ".dib", ".gif", ".webp", ".heic", ".heif", ".avif"}

_DATE_FOLDER_RE = re.compile(r"^\d{2}_\d{2}_\d{4}$")
_MFOLDER_RE = re.compile(r"^mFolder_(\d+)$")
_ROOT_DATE_RE = re.compile(r"(\d{2})[-_](\d{2})[-_](\d{4})")


@dataclass
class LegacyMeasurementCandidate:
    source_alias: str
    normalized_alias: str
    gallery_id: str
    measurement_date: str
    source_mfolder: Path
    source_sequence: int
    missing_files: List[str] = field(default_factory=list)
    mtime: float = 0.0

    @property
    def is_valid(self) -> bool:
        return not self.missing_files


@dataclass
class LegacyImportPlanItem:
    gallery_id: str
    source_alias: str
    normalized_alias: str
    measurement_date: str
    location: str
    source_mfolder: Path
    source_sequence: int


@dataclass
class LegacyImportResult:
    gallery_id: str
    source_alias: str
    measurement_date: str
    source_mfolder: str
    destination_mfolder: str = ""
    status: str = "planned"
    reason: str = ""
    missing_files: List[str] = field(default_factory=list)
    validation_errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gallery_id": self.gallery_id,
            "source_alias": self.source_alias,
            "measurement_date": self.measurement_date,
            "source_mfolder": self.source_mfolder,
            "destination_mfolder": self.destination_mfolder,
            "status": self.status,
            "reason": self.reason,
            "missing_files": list(self.missing_files),
            "validation_errors": list(self.validation_errors),
        }


@dataclass
class LegacyImportReport:
    source_root: str
    dry_run: bool
    location_hint: str
    results: List[LegacyImportResult] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def selected_count(self) -> int:
        return sum(
            1
            for item in self.results
            if item.status in {"planned", "imported", "already_imported", "metadata_appended"}
        )

    @property
    def imported_count(self) -> int:
        return sum(1 for item in self.results if item.status in {"imported", "metadata_appended"})

    @property
    def skipped_count(self) -> int:
        return sum(
            1
            for item in self.results
            if item.status.startswith("skipped")
        )

    @property
    def success(self) -> bool:
        return not self.errors and all(not item.validation_errors for item in self.results)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_root": self.source_root,
            "dry_run": self.dry_run,
            "location_hint": self.location_hint,
            "selected_count": self.selected_count,
            "imported_count": self.imported_count,
            "skipped_count": self.skipped_count,
            "errors": list(self.errors),
            "results": [item.to_dict() for item in self.results],
        }

    def to_text(self) -> str:
        lines = [
            f"source_root: {self.source_root}",
            f"mode: {'dry-run' if self.dry_run else 'apply'}",
            f"location_hint: {self.location_hint}",
            f"selected_count: {self.selected_count}",
            f"imported_count: {self.imported_count}",
            f"skipped_count: {self.skipped_count}",
            f"errors: {len(self.errors)}",
        ]
        for item in self.results:
            lines.append(
                f"- {item.gallery_id or '<unmapped>'}: {item.status} :: "
                f"{item.source_mfolder}"
                + (f" -> {item.destination_mfolder}" if item.destination_mfolder else "")
                + (f" ({item.reason})" if item.reason else "")
            )
            if item.validation_errors:
                lines.extend([f"  validation: {err}" for err in item.validation_errors])
        if self.errors:
            lines.extend([f"error: {err}" for err in self.errors])
        return "\n".join(lines)


@dataclass
class LegacyArchiveBackfillPlanItem:
    gallery_id: str
    measurement_date: str
    encounter_name: str
    imported_mfolder: Path
    raw_frame_path: Path
    archive_encounter_dir: Path
    legacy_source_mfolder: str


@dataclass
class LegacyArchiveBackfillResult:
    gallery_id: str
    measurement_date: str
    imported_mfolder: str
    archive_encounter_dir: str
    legacy_source_mfolder: str
    archive_image_path: str = ""
    status: str = "planned_backfill"
    reason: str = ""
    validation_errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gallery_id": self.gallery_id,
            "measurement_date": self.measurement_date,
            "imported_mfolder": self.imported_mfolder,
            "archive_encounter_dir": self.archive_encounter_dir,
            "legacy_source_mfolder": self.legacy_source_mfolder,
            "archive_image_path": self.archive_image_path,
            "status": self.status,
            "reason": self.reason,
            "validation_errors": list(self.validation_errors),
        }


@dataclass
class LegacyArchiveBackfillReport:
    source_root: str
    dry_run: bool
    results: List[LegacyArchiveBackfillResult] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def selected_count(self) -> int:
        return sum(
            1
            for item in self.results
            if item.status in {"planned_backfill", "backfilled", "already_archived"}
        )

    @property
    def backfilled_count(self) -> int:
        return sum(1 for item in self.results if item.status == "backfilled")

    @property
    def skipped_count(self) -> int:
        return sum(1 for item in self.results if item.status.startswith("skipped"))

    @property
    def success(self) -> bool:
        return not self.errors and all(not item.validation_errors for item in self.results)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_root": self.source_root,
            "dry_run": self.dry_run,
            "selected_count": self.selected_count,
            "backfilled_count": self.backfilled_count,
            "skipped_count": self.skipped_count,
            "errors": list(self.errors),
            "results": [item.to_dict() for item in self.results],
        }

    def to_text(self) -> str:
        lines = [
            f"source_root: {self.source_root}",
            f"mode: {'dry-run' if self.dry_run else 'apply'}",
            f"selected_count: {self.selected_count}",
            f"backfilled_count: {self.backfilled_count}",
            f"skipped_count: {self.skipped_count}",
            f"errors: {len(self.errors)}",
        ]
        for item in self.results:
            lines.append(
                f"- {item.gallery_id}: {item.status} :: {item.imported_mfolder}"
                + (f" -> {item.archive_image_path}" if item.archive_image_path else "")
                + (f" ({item.reason})" if item.reason else "")
            )
            if item.validation_errors:
                lines.extend([f"  validation: {err}" for err in item.validation_errors])
        if self.errors:
            lines.extend([f"error: {err}" for err in self.errors])
        return "\n".join(lines)


def default_legacy_source_root() -> Path:
    return get_measurements_root() / LEGACY_SOURCE_ROOT_NAME


def infer_measurement_date_from_root(source_root: Path) -> str:
    match = _ROOT_DATE_RE.search(source_root.name)
    if not match:
        raise ValueError(f"Could not infer measurement date from source root: {source_root}")
    mm, dd, yyyy = match.groups()
    return f"{mm}_{dd}_{yyyy}"


def infer_location_from_root(source_root: Path) -> str:
    name = source_root.name
    if name.startswith("tmp_"):
        name = name[4:]
    name = re.sub(r"[-_]merge_to_gallery$", "", name)
    match = _ROOT_DATE_RE.match(name)
    if not match:
        return name.replace("-", "_")
    mm, dd, yyyy = match.groups()
    rest = name[match.end():].lstrip("-_")
    rest = rest.replace("-", "_")
    date_part = f"{mm}-{dd}-{yyyy}"
    return f"{date_part}_{rest}" if rest else date_part


def _safe_mtime(path: Path) -> float:
    try:
        return float(path.stat().st_mtime)
    except Exception:
        return 0.0


def _mfolder_sequence(name: str) -> int:
    match = _MFOLDER_RE.match((name or "").strip())
    if not match:
        return -1
    try:
        return int(match.group(1))
    except Exception:
        return -1


def _normalize_alias(alias: str) -> str:
    return normalize_id_value(alias)


def _build_gallery_id_lookup(gallery_ids: Iterable[str]) -> Dict[str, str]:
    exact: Dict[str, str] = {}
    casefolded: Dict[str, List[str]] = {}
    for gallery_id in gallery_ids:
        normalized = normalize_id_value(gallery_id)
        if not normalized:
            continue
        exact[normalized] = gallery_id
        casefolded.setdefault(normalized.casefold(), []).append(gallery_id)

    out = dict(exact)
    for key, matches in casefolded.items():
        if len(matches) == 1:
            out.setdefault(key, matches[0])
    return out


def _lookup_gallery_id(alias: str, gallery_lookup: Dict[str, str]) -> str:
    normalized = _normalize_alias(alias)
    if not normalized:
        return ""
    return gallery_lookup.get(normalized) or gallery_lookup.get(normalized.casefold(), "")


def _discover_gallery_ids(gallery_rows: Sequence[Dict[str, str]], gallery_root: Optional[Path] = None) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()

    for row in gallery_rows:
        gallery_id = normalize_id_value(row.get("gallery_id", ""))
        if gallery_id and gallery_id not in seen:
            out.append(gallery_id)
            seen.add(gallery_id)

    root = gallery_root or ap.gallery_root()
    if root.exists():
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            gallery_id = normalize_id_value(child.name)
            if gallery_id and gallery_id not in seen:
                out.append(gallery_id)
                seen.add(gallery_id)

    return out


def _missing_required_files(mfolder: Path) -> List[str]:
    return [name for name in LEGACY_REQUIRED_FILENAMES if not (mfolder / name).exists()]


def _iter_legacy_mfolders(alias_dir: Path, default_date: str) -> Iterable[Tuple[str, Path]]:
    for child in sorted(alias_dir.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        if child.name.startswith("mFolder_"):
            yield default_date, child
            continue
        measurement_date = child.name if _DATE_FOLDER_RE.match(child.name) else default_date
        for nested in sorted(child.iterdir()):
            if nested.is_dir() and nested.name.startswith("mFolder_"):
                yield measurement_date, nested


def scan_legacy_candidates(
    source_root: Path,
    *,
    gallery_ids: Optional[Iterable[str]] = None,
) -> List[LegacyMeasurementCandidate]:
    source_root = Path(source_root)
    if not source_root.exists():
        raise FileNotFoundError(f"Legacy source root not found: {source_root}")

    default_date = infer_measurement_date_from_root(source_root)
    lookup = _build_gallery_id_lookup(gallery_ids or [])
    candidates: List[LegacyMeasurementCandidate] = []

    for alias_dir in sorted(source_root.iterdir()):
        if not alias_dir.is_dir() or alias_dir.name.startswith("."):
            continue
        gallery_id = _lookup_gallery_id(alias_dir.name, lookup)
        normalized_alias = _normalize_alias(alias_dir.name)
        for measurement_date, mfolder in _iter_legacy_mfolders(alias_dir, default_date):
            candidates.append(
                LegacyMeasurementCandidate(
                    source_alias=alias_dir.name,
                    normalized_alias=normalized_alias,
                    gallery_id=gallery_id,
                    measurement_date=measurement_date,
                    source_mfolder=mfolder,
                    source_sequence=_mfolder_sequence(mfolder.name),
                    missing_files=_missing_required_files(mfolder),
                    mtime=_safe_mtime(mfolder),
                )
            )

    return candidates


def _candidate_sort_key(candidate: LegacyMeasurementCandidate) -> Tuple[int, float, str]:
    return (
        candidate.source_sequence,
        candidate.mtime,
        str(candidate.source_mfolder),
    )


def _group_rows_by_gallery_id(gallery_rows: Sequence[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    grouped: Dict[str, List[Dict[str, str]]] = {}
    for row in gallery_rows:
        gallery_id = normalize_id_value(row.get("gallery_id", ""))
        if not gallery_id:
            continue
        grouped.setdefault(gallery_id, []).append(row)
    return grouped


def _resolve_location(
    gallery_id: str,
    rows_for_id: Sequence[Dict[str, str]],
    location_hint: str,
) -> str:
    exact_matches = [
        (row.get("location") or "").strip()
        for row in rows_for_id
        if (row.get("location") or "").strip() == location_hint
    ]
    if exact_matches:
        return exact_matches[-1]
    if location_hint:
        return location_hint
    locations = [(row.get("location") or "").strip() for row in rows_for_id if (row.get("location") or "").strip()]
    return locations[-1] if locations else ""


def build_import_plan(
    source_root: Path,
    *,
    gallery_rows: Optional[Sequence[Dict[str, str]]] = None,
    gallery_ids: Optional[Iterable[str]] = None,
    location_hint: Optional[str] = None,
) -> Tuple[List[LegacyImportPlanItem], List[LegacyImportResult]]:
    source_root = Path(source_root)
    location_hint = location_hint or infer_location_from_root(source_root)
    gallery_rows = list(gallery_rows or [])
    discovered_gallery_ids = list(gallery_ids or _discover_gallery_ids(gallery_rows))
    candidates = scan_legacy_candidates(source_root, gallery_ids=discovered_gallery_ids)
    grouped_rows = _group_rows_by_gallery_id(gallery_rows)

    results: List[LegacyImportResult] = []
    plan_items: List[LegacyImportPlanItem] = []

    unmapped = [candidate for candidate in candidates if not candidate.gallery_id]
    for candidate in unmapped:
        results.append(
            LegacyImportResult(
                gallery_id="",
                source_alias=candidate.source_alias,
                measurement_date=candidate.measurement_date,
                source_mfolder=str(candidate.source_mfolder),
                status="skipped_missing_gallery",
                reason=f"No gallery_id match for alias '{candidate.normalized_alias}'",
                missing_files=list(candidate.missing_files),
            )
        )

    grouped_candidates: Dict[Tuple[str, str], List[LegacyMeasurementCandidate]] = {}
    for candidate in candidates:
        if not candidate.gallery_id:
            continue
        key = (candidate.gallery_id, candidate.measurement_date)
        grouped_candidates.setdefault(key, []).append(candidate)

    for (gallery_id, measurement_date) in sorted(grouped_candidates.keys()):
        group = grouped_candidates[(gallery_id, measurement_date)]
        valid = [candidate for candidate in group if candidate.is_valid]
        invalid = [candidate for candidate in group if not candidate.is_valid]
        for candidate in invalid:
            results.append(
                LegacyImportResult(
                    gallery_id=gallery_id,
                    source_alias=candidate.source_alias,
                    measurement_date=measurement_date,
                    source_mfolder=str(candidate.source_mfolder),
                    status="skipped_invalid_bundle",
                    reason="Missing required legacy files",
                    missing_files=list(candidate.missing_files),
                )
            )

        if not valid:
            continue

        selected = max(valid, key=_candidate_sort_key)
        location = _resolve_location(
            gallery_id=gallery_id,
            rows_for_id=grouped_rows.get(gallery_id, []),
            location_hint=location_hint,
        )
        plan_items.append(
            LegacyImportPlanItem(
                gallery_id=gallery_id,
                source_alias=selected.source_alias,
                normalized_alias=selected.normalized_alias,
                measurement_date=measurement_date,
                location=location,
                source_mfolder=selected.source_mfolder,
                source_sequence=selected.source_sequence,
            )
        )
        for candidate in valid:
            if candidate.source_mfolder == selected.source_mfolder:
                continue
            results.append(
                LegacyImportResult(
                    gallery_id=gallery_id,
                    source_alias=candidate.source_alias,
                    measurement_date=measurement_date,
                    source_mfolder=str(candidate.source_mfolder),
                    status="skipped_older_same_day",
                    reason=(
                        f"Kept newest valid same-day bundle "
                        f"({selected.source_mfolder.name}) for {gallery_id}"
                    ),
                )
            )

    plan_items.sort(key=lambda item: (_date_sort_key(item.measurement_date), item.gallery_id, item.source_sequence))
    return plan_items, results


def _date_sort_key(date_str: str) -> Tuple[int, int, int, str]:
    try:
        parsed = datetime.strptime(date_str, "%m_%d_%Y")
        return (parsed.year, parsed.month, parsed.day, date_str)
    except Exception:
        return (0, 0, 0, date_str)


def _parse_measurement_date(date_str: str) -> Optional[date]:
    for fmt in ("%m_%d_%Y", "%m_%d_%y"):
        try:
            return datetime.strptime((date_str or "").strip(), fmt).date()
        except Exception:
            continue
    return None


def _encounter_name_from_measurement_date(date_str: str) -> str:
    parsed = _parse_measurement_date(date_str)
    if parsed is None:
        raise ValueError(f"Unsupported measurement date format: {date_str}")
    return ensure_encounter_name(parsed.year, parsed.month, parsed.day)


def _select_historical_row(rows_for_id: Sequence[Dict[str, str]], location: str) -> Dict[str, str]:
    for row in reversed(list(rows_for_id)):
        if (row.get("location") or "").strip() == location:
            return row
    return {}


def _combine_base_metadata_rows(
    header: Sequence[str],
    current_latest: Optional[Dict[str, str]],
    historical_row: Optional[Dict[str, str]],
) -> Dict[str, str]:
    combined = {column: "" for column in header}
    for source in (current_latest or {}, historical_row or {}):
        for column in header:
            value = (source.get(column) or "")
            if value:
                combined[column] = value
    return combined


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=4)


def _copy_bundle(source_mfolder: Path, destination_mfolder: Path) -> None:
    destination_mfolder.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        source_mfolder,
        destination_mfolder,
        ignore=shutil.ignore_patterns(".DS_Store"),
    )


def _patch_imported_bundle(
    *,
    source_mfolder: Path,
    destination_mfolder: Path,
    gallery_id: str,
    location: str,
) -> Dict[str, Any]:
    detection_path = destination_mfolder / "corrected_detection.json"
    morph_path = destination_mfolder / "morphometrics.json"
    detection = _read_json(detection_path)
    morph = _read_json(morph_path)

    mask_path = destination_mfolder / "corrected_mask.png"
    object_path = destination_mfolder / "corrected_object.png"
    raw_path = destination_mfolder / "raw_frame.png"
    combined_path = destination_mfolder / "checkerboard_with_object.png"

    mm_per_pixel = morph.get("mm_per_pixel")
    if mm_per_pixel in ("", None):
        mm_per_pixel = detection.get("mm_per_pixel")

    detection["location"] = location
    detection["identity_type"] = "gallery"
    detection["identity_id"] = gallery_id
    detection[LEGACY_PROVENANCE_KEY] = str(source_mfolder)
    detection["mask_path"] = str(mask_path)
    detection["object_path"] = str(object_path)
    detection["raw_frame_path"] = str(raw_path)
    if combined_path.exists():
        detection["combined_image_path"] = str(combined_path)
    if detection.get("class_name") in ("", None):
        detection["class_name"] = "Pycnopodia_helianthoides"
    if mm_per_pixel not in ("", None):
        detection["mm_per_pixel"] = mm_per_pixel

    morph.pop("arm_lengths_mm", None)
    morph["location"] = location
    morph["identity_type"] = "gallery"
    morph["identity_id"] = gallery_id
    morph[LEGACY_PROVENANCE_KEY] = str(source_mfolder)
    morph["arm_rotation"] = int(morph.get("arm_rotation") or 0)
    morph["user_notes"] = str(morph.get("user_notes") or "")
    if isinstance(morph.get("user_initials"), str):
        morph["user_initials"] = morph["user_initials"].strip().upper()
    if mm_per_pixel not in ("", None):
        morph["mm_per_pixel"] = mm_per_pixel

    _write_json(detection_path, detection)
    _write_json(morph_path, morph)
    return morph


def _next_available_mfolder(date_dir: Path, preferred_name: str) -> Path:
    preferred = date_dir / preferred_name
    if not preferred.exists():
        return preferred
    existing_numbers: List[int] = []
    for child in date_dir.iterdir():
        if not child.is_dir():
            continue
        seq = _mfolder_sequence(child.name)
        if seq >= 0:
            existing_numbers.append(seq)
    next_number = max(existing_numbers, default=0) + 1
    return date_dir / f"mFolder_{next_number}"


def _bundle_provenance(path: Path) -> str:
    for json_name in ("morphometrics.json", "corrected_detection.json"):
        candidate = path / json_name
        if not candidate.exists():
            continue
        try:
            data = _read_json(candidate)
        except Exception:
            continue
        provenance = (data.get(LEGACY_PROVENANCE_KEY) or "").strip()
        if provenance:
            return provenance
    return ""


def _find_matching_archive_image(archive_encounter_dir: Path, raw_frame_path: Path) -> Optional[Path]:
    if not archive_encounter_dir.exists() or not raw_frame_path.exists():
        return None
    for candidate in sorted(archive_encounter_dir.iterdir()):
        if not candidate.is_file():
            continue
        if candidate.suffix.lower() not in ARCHIVE_IMAGE_EXTS:
            continue
        try:
            if filecmp.cmp(raw_frame_path, candidate, shallow=False):
                return candidate
        except Exception:
            continue
    return None


def _ensure_archive_image_for_measurement(
    *,
    gallery_id: str,
    measurement_date: str,
    raw_frame_path: Path,
    gallery_root: Path,
    dry_run: bool,
) -> Tuple[str, Path, str]:
    if not raw_frame_path.exists():
        raise FileNotFoundError(f"Missing raw_frame.png for archive copy: {raw_frame_path}")

    encounter_name = _encounter_name_from_measurement_date(measurement_date)
    archive_encounter_dir = gallery_root / gallery_id / encounter_name
    existing_match = _find_matching_archive_image(archive_encounter_dir, raw_frame_path)
    if existing_match is not None:
        return "already_archived", existing_match, ""

    planned_path = archive_encounter_dir / raw_frame_path.name
    if dry_run:
        return "planned_backfill", planned_path, "Ready to backfill archive image"

    parsed_date = _parse_measurement_date(measurement_date)
    report = place_images(
        gallery_root,
        gallery_id,
        encounter_name,
        [raw_frame_path],
        move=False,
        observation_date=parsed_date,
    )
    if report.errors:
        raise ValueError("; ".join(report.errors))
    if not report.ops:
        raise ValueError(f"No archive image was created for {raw_frame_path}")
    return "backfilled", report.ops[0].dest, ""


def _find_existing_import(date_dir: Path, source_mfolder: Path) -> Optional[Path]:
    if not date_dir.exists():
        return None
    source_value = str(source_mfolder)
    for child in sorted(date_dir.iterdir()):
        if not child.is_dir() or not child.name.startswith("mFolder_"):
            continue
        if _bundle_provenance(child) == source_value:
            return child
    return None


def _metadata_row_exists(rows_for_id: Sequence[Dict[str, str]], destination_mfolder: Path) -> bool:
    destination_value = str(destination_mfolder)
    for row in rows_for_id:
        if (row.get("morph_source_folder") or "").strip() == destination_value:
            return True
    return False


def _build_metadata_row(
    *,
    gallery_id: str,
    location: str,
    destination_mfolder: Path,
    morph_fields: Dict[str, str],
    header: Sequence[str],
    current_latest: Optional[Dict[str, str]],
    historical_row: Optional[Dict[str, str]],
) -> Dict[str, str]:
    row = _combine_base_metadata_rows(header, current_latest, historical_row)
    row["gallery_id"] = gallery_id
    row["location"] = location
    for key, value in morph_fields.items():
        row[key] = value
    row["morph_source_folder"] = str(destination_mfolder)
    return row


def _validate_archive_image_visibility(gallery_id: str, archive_image_path: Path) -> List[str]:
    errors: List[str] = []
    if not archive_image_path.exists():
        errors.append(f"Archive image missing: {archive_image_path}")
        return errors

    listed = {str(path) for path in list_image_files("Gallery", gallery_id)}
    if str(archive_image_path) not in listed:
        errors.append(f"Archive image is not discoverable via list_image_files(): {archive_image_path}")
    return errors


def _validate_imported_bundle(
    *,
    destination_mfolder: Path,
    measurements_root: Path,
    gallery_id: str,
) -> List[str]:
    errors: List[str] = []
    for filename in LEGACY_REQUIRED_FILENAMES:
        if not (destination_mfolder / filename).exists():
            errors.append(f"Missing required imported file: {filename}")

    listed = {str(path) for path in list_mfolders(measurements_root, "gallery", gallery_id)}
    if str(destination_mfolder) not in listed:
        errors.append("Imported mFolder is not discoverable via list_mfolders()")

    morph = load_morphometrics_from_mfolder(destination_mfolder) or {}
    if (morph.get("identity_type") or "").strip().lower() != "gallery":
        errors.append("Imported morphometrics.json is missing identity_type=gallery")
    if (morph.get("identity_id") or "").strip() != gallery_id:
        errors.append(f"Imported morphometrics.json is missing identity_id={gallery_id}")

    return errors


def build_archive_backfill_plan(
    source_root: Path = default_legacy_source_root(),
    *,
    measurements_root: Optional[Path] = None,
    gallery_root: Optional[Path] = None,
) -> List[LegacyArchiveBackfillPlanItem]:
    source_root = Path(source_root)
    measurements_root = Path(measurements_root) if measurements_root else get_measurements_root()
    gallery_root = Path(gallery_root) if gallery_root else ap.root_for("Gallery")

    plan_items: List[LegacyArchiveBackfillPlanItem] = []
    for mfolder in list_mfolders(measurements_root, "gallery"):
        provenance = _bundle_provenance(mfolder)
        if not provenance:
            continue
        provenance_path = Path(provenance)
        if str(provenance_path) != str(source_root) and not str(provenance_path).startswith(str(source_root) + "/"):
            continue

        raw_frame_path = mfolder / "raw_frame.png"
        try:
            gallery_id = mfolder.parents[1].name
            measurement_date = mfolder.parents[0].name
        except IndexError:
            continue

        encounter_name = _encounter_name_from_measurement_date(measurement_date)
        plan_items.append(
            LegacyArchiveBackfillPlanItem(
                gallery_id=gallery_id,
                measurement_date=measurement_date,
                encounter_name=encounter_name,
                imported_mfolder=mfolder,
                raw_frame_path=raw_frame_path,
                archive_encounter_dir=gallery_root / gallery_id / encounter_name,
                legacy_source_mfolder=provenance,
            )
        )

    plan_items.sort(key=lambda item: (_date_sort_key(item.measurement_date), item.gallery_id, str(item.imported_mfolder)))
    return plan_items


def backfill_legacy_gallery_archive_images(
    source_root: Path = default_legacy_source_root(),
    *,
    dry_run: bool = True,
    measurements_root: Optional[Path] = None,
    gallery_root: Optional[Path] = None,
) -> LegacyArchiveBackfillReport:
    source_root = Path(source_root)
    gallery_root = Path(gallery_root) if gallery_root else ap.root_for("Gallery")
    plan_items = build_archive_backfill_plan(
        source_root=source_root,
        measurements_root=measurements_root,
        gallery_root=gallery_root,
    )
    report = LegacyArchiveBackfillReport(source_root=str(source_root), dry_run=dry_run)

    for item in plan_items:
        if not item.raw_frame_path.exists():
            report.results.append(
                LegacyArchiveBackfillResult(
                    gallery_id=item.gallery_id,
                    measurement_date=item.measurement_date,
                    imported_mfolder=str(item.imported_mfolder),
                    archive_encounter_dir=str(item.archive_encounter_dir),
                    legacy_source_mfolder=item.legacy_source_mfolder,
                    status="skipped_missing_raw_frame",
                    reason="Imported measurement bundle is missing raw_frame.png",
                )
            )
            continue

        try:
            status, archive_image_path, reason = _ensure_archive_image_for_measurement(
                gallery_id=item.gallery_id,
                measurement_date=item.measurement_date,
                raw_frame_path=item.raw_frame_path,
                gallery_root=gallery_root,
                dry_run=dry_run,
            )
            validation_errors: List[str] = []
            if not dry_run:
                validation_errors = _validate_archive_image_visibility(item.gallery_id, archive_image_path)
            report.results.append(
                LegacyArchiveBackfillResult(
                    gallery_id=item.gallery_id,
                    measurement_date=item.measurement_date,
                    imported_mfolder=str(item.imported_mfolder),
                    archive_encounter_dir=str(item.archive_encounter_dir),
                    legacy_source_mfolder=item.legacy_source_mfolder,
                    archive_image_path=str(archive_image_path),
                    status=status,
                    reason=reason,
                    validation_errors=validation_errors,
                )
            )
        except Exception as exc:
            message = f"{item.gallery_id}: failed archive backfill for {item.imported_mfolder}: {exc}"
            log.exception(message)
            report.errors.append(message)
            report.results.append(
                LegacyArchiveBackfillResult(
                    gallery_id=item.gallery_id,
                    measurement_date=item.measurement_date,
                    imported_mfolder=str(item.imported_mfolder),
                    archive_encounter_dir=str(item.archive_encounter_dir),
                    legacy_source_mfolder=item.legacy_source_mfolder,
                    status="error",
                    reason=str(exc),
                )
            )

    return report


def import_legacy_gallery_measurements(
    source_root: Path = default_legacy_source_root(),
    *,
    dry_run: bool = True,
    location_hint: Optional[str] = None,
    measurements_root: Optional[Path] = None,
    gallery_csv_path: Optional[Path] = None,
) -> LegacyImportReport:
    source_root = Path(source_root)
    measurements_root = Path(measurements_root) if measurements_root else get_measurements_root()
    location_hint = location_hint or infer_location_from_root(source_root)
    gallery_csv_path = Path(gallery_csv_path) if gallery_csv_path else ap.metadata_csv_for("Gallery")[0]
    gallery_root = ap.root_for("Gallery")
    _, gallery_header = ap.metadata_csv_for("Gallery")
    gallery_rows = read_rows(gallery_csv_path)

    plan_items, preflight_results = build_import_plan(
        source_root,
        gallery_rows=gallery_rows,
        location_hint=location_hint,
    )
    report = LegacyImportReport(
        source_root=str(source_root),
        dry_run=dry_run,
        location_hint=location_hint,
        results=list(preflight_results),
    )

    rows_by_id = _group_rows_by_gallery_id(gallery_rows)
    latest_by_id = last_row_per_id(gallery_rows, "gallery_id")

    for item in plan_items:
        date_dir = measurements_root / "gallery" / item.gallery_id / item.measurement_date
        existing_import = _find_existing_import(date_dir, item.source_mfolder)
        destination_mfolder = existing_import or _next_available_mfolder(date_dir, item.source_mfolder.name)

        if dry_run:
            report.results.append(
                LegacyImportResult(
                    gallery_id=item.gallery_id,
                    source_alias=item.source_alias,
                    measurement_date=item.measurement_date,
                    source_mfolder=str(item.source_mfolder),
                    destination_mfolder=str(destination_mfolder),
                    status="planned",
                    reason="Ready to import and copy archive image",
                )
            )
            continue

        try:
            if existing_import is None:
                _copy_bundle(item.source_mfolder, destination_mfolder)
                morph = _patch_imported_bundle(
                    source_mfolder=item.source_mfolder,
                    destination_mfolder=destination_mfolder,
                    gallery_id=item.gallery_id,
                    location=item.location,
                )
            else:
                morph = load_morphometrics_from_mfolder(destination_mfolder) or {}

            morph_fields = extract_starboard_fields(morph)
            if not morph_fields:
                raise ValueError(f"No morph_* fields could be derived from {destination_mfolder}")

            rows_for_id = rows_by_id.get(item.gallery_id, [])
            current_latest = latest_by_id.get(item.gallery_id, {})
            historical_row = _select_historical_row(rows_for_id, item.location)

            result_status = "already_imported" if existing_import else "imported"
            result_reason = "Measurement bundle already existed" if existing_import else ""

            if not _metadata_row_exists(rows_for_id, destination_mfolder):
                new_row = _build_metadata_row(
                    gallery_id=item.gallery_id,
                    location=item.location,
                    destination_mfolder=destination_mfolder,
                    morph_fields=morph_fields,
                    header=gallery_header,
                    current_latest=current_latest,
                    historical_row=historical_row,
                )
                append_row(gallery_csv_path, gallery_header, new_row)
                record_morphometric_import(
                    gallery_id=item.gallery_id,
                    old_values=current_latest,
                    new_values=new_row,
                    mfolder_path=str(destination_mfolder),
                )
                gallery_rows.append(new_row)
                rows_by_id.setdefault(item.gallery_id, []).append(new_row)
                latest_by_id[item.gallery_id] = new_row
                if result_status == "already_imported":
                    result_status = "metadata_appended"
                    result_reason = "Existing imported bundle found; metadata row appended"
            raw_frame_path = destination_mfolder / "raw_frame.png"
            archive_validation_errors: List[str] = []
            if raw_frame_path.exists():
                archive_status, archive_image_path, archive_reason = _ensure_archive_image_for_measurement(
                    gallery_id=item.gallery_id,
                    measurement_date=item.measurement_date,
                    raw_frame_path=raw_frame_path,
                    gallery_root=gallery_root,
                    dry_run=False,
                )
                archive_validation_errors = _validate_archive_image_visibility(item.gallery_id, archive_image_path)
                if archive_status == "backfilled" and existing_import:
                    result_reason = "; ".join(filter(None, [result_reason, "Archive image backfilled"]))
                elif archive_status == "already_archived" and not result_reason:
                    result_reason = "Archive image already present"
            else:
                archive_validation_errors.append(
                    f"Imported bundle is missing raw_frame.png for archive copy: {destination_mfolder}"
                )
            validation_errors = _validate_imported_bundle(
                destination_mfolder=destination_mfolder,
                measurements_root=measurements_root,
                gallery_id=item.gallery_id,
            )
            validation_errors.extend(archive_validation_errors)
            report.results.append(
                LegacyImportResult(
                    gallery_id=item.gallery_id,
                    source_alias=item.source_alias,
                    measurement_date=item.measurement_date,
                    source_mfolder=str(item.source_mfolder),
                    destination_mfolder=str(destination_mfolder),
                    status=result_status,
                    reason=result_reason,
                    validation_errors=validation_errors,
                )
            )
        except Exception as exc:
            message = f"{item.gallery_id}: failed to import {item.source_mfolder}: {exc}"
            log.exception(message)
            report.errors.append(message)
            report.results.append(
                LegacyImportResult(
                    gallery_id=item.gallery_id,
                    source_alias=item.source_alias,
                    measurement_date=item.measurement_date,
                    source_mfolder=str(item.source_mfolder),
                    destination_mfolder=str(destination_mfolder),
                    status="error",
                    reason=str(exc),
                )
            )

    return report


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="One-off importer for deprecated legacy morphometric gallery measurements.",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=default_legacy_source_root(),
        help="Legacy measurement tree to import.",
    )
    parser.add_argument(
        "--location",
        default=None,
        help="Override the inferred historical location label.",
    )
    parser.add_argument(
        "--measurements-root",
        type=Path,
        default=None,
        help="Override the destination measurements root.",
    )
    parser.add_argument(
        "--gallery-csv",
        type=Path,
        default=None,
        help="Override the destination gallery metadata CSV path.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Perform the import. Default is dry-run only.",
    )
    parser.add_argument(
        "--backfill-archive-images",
        action="store_true",
        help="Backfill archive raw_frame images for already-imported legacy measurements.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the report as JSON instead of human-readable text.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.backfill_archive_images:
        report = backfill_legacy_gallery_archive_images(
            source_root=args.source_root,
            dry_run=not args.apply,
            measurements_root=args.measurements_root,
        )
    else:
        report = import_legacy_gallery_measurements(
            source_root=args.source_root,
            dry_run=not args.apply,
            location_hint=args.location,
            measurements_root=args.measurements_root,
            gallery_csv_path=args.gallery_csv,
        )

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(report.to_text())

    return 0 if report.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
