from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, date
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import re

from src.morphometric import get_measurements_root
from src.morphometric.data_bridge import (
    calculate_tip_to_tip_diameter,
    list_mfolders,
    load_morphometrics_from_mfolder,
)

log = logging.getLogger("starBoard.data.morphometric_analytics")


# Canonical size metrics exposed by the morphometric integration.
MORPHOMETRIC_METRIC_FIELDS: List[str] = [
    "morph_area_mm2",
    "morph_major_axis_mm",
    "morph_minor_axis_mm",
    "morph_mean_arm_length_mm",
    "morph_max_arm_length_mm",
    "morph_tip_to_tip_mm",
    "morph_num_arms",
]

MORPHOMETRIC_METRIC_LABELS: Dict[str, str] = {
    "morph_area_mm2": "Area (mm^2)",
    "morph_major_axis_mm": "Major axis (mm)",
    "morph_minor_axis_mm": "Minor axis (mm)",
    "morph_mean_arm_length_mm": "Mean arm length (mm)",
    "morph_max_arm_length_mm": "Max arm length (mm)",
    "morph_tip_to_tip_mm": "Tip-to-tip (mm)",
    "morph_num_arms": "Arms detected",
}


@dataclass
class MorphometricAnalyticsDataset:
    """Tabular morphometric records ready for analytics visualizations.

    - rows_raw: all records discovered from mFolders
    - rows: per-day deduped records where latest measurement wins
    """

    rows_raw: List[Dict[str, Any]]
    rows: List[Dict[str, Any]]
    metric_fields: List[str]


def _as_float(value: Any) -> Optional[float]:
    if value in ("", None):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _as_int(value: Any) -> Optional[int]:
    if value in ("", None):
        return None
    try:
        return int(value)
    except Exception:
        try:
            return int(float(value))
        except Exception:
            return None


def _parse_measurement_day(date_label: str, mfolder: Path) -> tuple[str, Optional[date]]:
    for fmt in ("%m_%d_%Y", "%m_%d_%y", "%Y-%m-%d"):
        try:
            parsed = datetime.strptime((date_label or "").strip(), fmt).date()
            return parsed.isoformat(), parsed
        except Exception:
            continue

    # Fallback to file modification day if folder naming is non-standard.
    try:
        parsed = datetime.fromtimestamp(mfolder.stat().st_mtime).date()
        return parsed.isoformat(), parsed
    except Exception:
        return "", None


def _parse_iso_day(value: str) -> Optional[date]:
    s = (value or "").strip()
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except Exception:
        return None


def _parse_mfolder_sequence(name: str) -> int:
    """
    Extract numeric suffix from `mFolder_N`.
    Returns -1 if unavailable.
    """
    m = re.match(r"^mFolder_(\d+)$", (name or "").strip())
    if not m:
        return -1
    try:
        return int(m.group(1))
    except Exception:
        return -1


def _read_location_fallback(mfolder: Path) -> str:
    detection_path = mfolder / "corrected_detection.json"
    if not detection_path.exists():
        return ""
    try:
        with detection_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return str(data.get("location", "")).strip()
    except Exception:
        return ""


def _extract_path_context(mfolder: Path) -> tuple[str, str, str]:
    """
    mFolder path shape:
      measurements/<identity_type>/<identity_id>/<mm_dd_yyyy>/mFolder_N
    """
    identity_type = ""
    identity_id = ""
    date_label = ""
    try:
        identity_type = mfolder.parents[2].name.lower()
        identity_id = mfolder.parents[1].name
        date_label = mfolder.parents[0].name
    except Exception:
        pass
    return identity_type, identity_id, date_label


def _arm_lengths(arm_data: Any) -> List[float]:
    out: List[float] = []
    if not isinstance(arm_data, list):
        return out
    for arm in arm_data:
        if not isinstance(arm, (list, tuple)) or len(arm) < 4:
            continue
        length_mm = _as_float(arm[3])
        if length_mm is not None:
            out.append(length_mm)
    return out


def _build_row(mfolder: Path) -> Optional[Dict[str, Any]]:
    identity_type, identity_id, date_label = _extract_path_context(mfolder)
    if not identity_id:
        return None

    morph = load_morphometrics_from_mfolder(mfolder)
    if not morph:
        return None

    arm_data = morph.get("arm_data", [])
    lengths = _arm_lengths(arm_data)

    mean_arm = _as_float(morph.get("mean_arm_length_mm"))
    if mean_arm is None and lengths:
        mean_arm = sum(lengths) / len(lengths)

    max_arm = _as_float(morph.get("max_arm_length_mm"))
    if max_arm is None and lengths:
        max_arm = max(lengths)

    tip_to_tip = _as_float(morph.get("tip_to_tip_mm"))
    if tip_to_tip is None and isinstance(arm_data, list) and arm_data:
        tip_to_tip = _as_float(calculate_tip_to_tip_diameter(arm_data))

    num_arms = _as_int(morph.get("num_arms"))
    if num_arms is None and lengths:
        num_arms = len(lengths)

    measurement_day, _ = _parse_measurement_day(date_label, mfolder)
    location = str(morph.get("location", "")).strip() or _read_location_fallback(mfolder)
    sequence = _parse_mfolder_sequence(mfolder.name)
    try:
        mtime = float(mfolder.stat().st_mtime)
    except Exception:
        mtime = 0.0

    row: Dict[str, Any] = {
        "identity_type": identity_type,
        "identity_id": identity_id,
        "identity_label": f"{identity_type}:{identity_id}" if identity_type else identity_id,
        "location": location,
        "measurement_day": measurement_day,
        "measurement_day_raw": date_label,
        "mfolder": str(mfolder),
        "mfolder_name": mfolder.name,
        "measurement_sequence": sequence,
        "measurement_mtime": mtime,
        "morph_area_mm2": _as_float(morph.get("area_mm2")),
        "morph_major_axis_mm": _as_float(morph.get("major_axis_mm")),
        "morph_minor_axis_mm": _as_float(morph.get("minor_axis_mm")),
        "morph_mean_arm_length_mm": mean_arm,
        "morph_max_arm_length_mm": max_arm,
        "morph_tip_to_tip_mm": tip_to_tip,
        "morph_num_arms": num_arms,
    }
    return row


def _row_sort_key(row: Dict[str, Any]) -> tuple[str, str, str, int, float, str]:
    return (
        str(row.get("measurement_day") or ""),
        str(row.get("identity_type") or ""),
        str(row.get("identity_id") or ""),
        int(row.get("measurement_sequence") or -1),
        float(row.get("measurement_mtime") or 0.0),
        str(row.get("mfolder") or ""),
    )


def _is_newer_row(candidate: Dict[str, Any], current: Dict[str, Any]) -> bool:
    """
    Compare records from the same identity/day and decide whether candidate is newer.

    Primary: `measurement_sequence` (mFolder_N)
    Secondary: folder mtime
    Tertiary: mfolder path lexicographic
    """
    cand_seq = int(candidate.get("measurement_sequence") or -1)
    curr_seq = int(current.get("measurement_sequence") or -1)
    if cand_seq != curr_seq:
        return cand_seq > curr_seq

    cand_mtime = float(candidate.get("measurement_mtime") or 0.0)
    curr_mtime = float(current.get("measurement_mtime") or 0.0)
    if cand_mtime != curr_mtime:
        return cand_mtime > curr_mtime

    return str(candidate.get("mfolder") or "") > str(current.get("mfolder") or "")


def dedupe_last_measurement_per_day(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Deduplicate records by `(identity_type, identity_id, measurement_day)` keeping newest.
    """
    winners: Dict[tuple[str, str, str], Dict[str, Any]] = {}
    for row in rows:
        day_key = str(row.get("measurement_day") or row.get("measurement_day_raw") or row.get("mfolder") or "")
        key = (
            str(row.get("identity_type") or ""),
            str(row.get("identity_id") or ""),
            day_key,
        )
        existing = winners.get(key)
        if existing is None or _is_newer_row(row, existing):
            winners[key] = row

    out = list(winners.values())
    out.sort(key=_row_sort_key)
    return out


def filter_morphometric_rows(
    rows: List[Dict[str, Any]],
    *,
    scope: str = "all",
    identity_labels: Optional[List[str]] = None,
    identity_ids: Optional[List[str]] = None,
    locations: Optional[List[str]] = None,
    start_day: Optional[str] = None,
    end_day: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Apply common filters for morphometric analytics.
    """
    wanted_scope = (scope or "all").strip().lower()
    wanted_labels = {s.strip() for s in (identity_labels or []) if s and s.strip()}
    wanted_ids = {s.strip() for s in (identity_ids or []) if s and s.strip()}
    wanted_locations = {s.strip() for s in (locations or []) if s and s.strip()}

    start_dt = _parse_iso_day(start_day or "")
    end_dt = _parse_iso_day(end_day or "")

    out: List[Dict[str, Any]] = []
    for row in rows:
        if wanted_scope != "all":
            if str(row.get("identity_type") or "").strip().lower() != wanted_scope:
                continue

        if wanted_labels:
            if str(row.get("identity_label") or "") not in wanted_labels:
                continue

        if wanted_ids:
            if str(row.get("identity_id") or "") not in wanted_ids:
                continue

        if wanted_locations:
            if str(row.get("location") or "").strip() not in wanted_locations:
                continue

        if start_dt is not None or end_dt is not None:
            row_day = _parse_iso_day(str(row.get("measurement_day") or ""))
            if row_day is None:
                continue
            if start_dt is not None and row_day < start_dt:
                continue
            if end_dt is not None and row_day > end_dt:
                continue

        out.append(row)

    out.sort(key=_row_sort_key)
    return out


def available_locations(rows: List[Dict[str, Any]], *, scope: str = "all") -> List[str]:
    filtered = filter_morphometric_rows(rows, scope=scope)
    vals = sorted({str(r.get("location") or "").strip() for r in filtered if str(r.get("location") or "").strip()})
    return vals


def available_identity_labels(
    rows: List[Dict[str, Any]],
    *,
    scope: str = "all",
    location: Optional[str] = None,
) -> List[str]:
    locations = [location] if location and location.strip() and location.strip().lower() != "all" else None
    filtered = filter_morphometric_rows(rows, scope=scope, locations=locations)
    vals = sorted({str(r.get("identity_label") or "").strip() for r in filtered if str(r.get("identity_label") or "").strip()})
    return vals


def build_morphometric_analytics_dataset() -> MorphometricAnalyticsDataset:
    """
    Build analytics-ready morphometric rows from saved mFolders.
    """
    rows_raw: List[Dict[str, Any]] = []
    try:
        root = get_measurements_root()
        if not root.exists():
            return MorphometricAnalyticsDataset(
                rows_raw=[],
                rows=[],
                metric_fields=MORPHOMETRIC_METRIC_FIELDS.copy(),
            )

        for mfolder in list_mfolders(root):
            row = _build_row(Path(mfolder))
            if row is not None:
                rows_raw.append(row)
    except Exception as e:
        log.warning("Failed to build morphometric analytics dataset: %s", e)

    rows_raw.sort(key=_row_sort_key)
    rows = dedupe_last_measurement_per_day(rows_raw)
    return MorphometricAnalyticsDataset(
        rows_raw=rows_raw,
        rows=rows,
        metric_fields=MORPHOMETRIC_METRIC_FIELDS.copy(),
    )

