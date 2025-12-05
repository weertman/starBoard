# src/data/compare_labels.py
from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

from src.data.archive_paths import queries_root
from src.data.csv_io import (
    ensure_header, append_row, read_rows, last_row_per_id,
    normalize_id_value
)

LABELS_HEADER = ["query_id", "gallery_id", "verdict", "notes", "updated_utc"]

def _csv_path_for_query(query_id: str) -> Path:
    return queries_root(prefer_new=True) / query_id / "_second_order_labels.csv"

def load_latest_map_for_query(query_id: str) -> Dict[str, Dict[str, str]]:
    path = _csv_path_for_query(query_id)
    rows = read_rows(path) if path.exists() else []
    rows = [r for r in rows if normalize_id_value(r.get("query_id", "")) == normalize_id_value(query_id)]
    latest = last_row_per_id(rows, "gallery_id")
    return latest

def get_label_for_pair(query_id: str, gallery_id: str) -> Optional[Dict[str, str]]:
    latest = load_latest_map_for_query(query_id)
    row = latest.get(normalize_id_value(gallery_id))
    return dict(row) if row else None

def save_label_for_pair(query_id: str, gallery_id: str, verdict: str, notes: str) -> None:
    path = _csv_path_for_query(query_id)
    ensure_header(path, LABELS_HEADER)
    obj = {
        "query_id": query_id,
        "gallery_id": gallery_id,
        "verdict": (verdict or "").strip().lower(),
        "notes": notes or "",
        "updated_utc": datetime.utcnow().isoformat() + "Z",
    }
    append_row(path, LABELS_HEADER, obj)
