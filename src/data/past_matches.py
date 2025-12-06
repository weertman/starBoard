# src/data/past_matches.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional
from collections import Counter, defaultdict
import csv

from src.data.archive_paths import (
    roots_for_read,
    metadata_csv_paths_for_read,
    id_column_name,
    QUERIES_HEADER,
    GALLERY_HEADER,
    archive_root,
)
from src.data.csv_io import (
    read_rows_multi, read_rows, last_row_per_id,
    normalize_id_value, ensure_header,
)
from src.data.compare_labels import LABELS_HEADER  # ["query_id","gallery_id","verdict","notes","updated_utc"]

# ---------------------------- types ----------------------------
Row = Dict[str, str]
Verdict = str  # "yes" | "maybe" | "no"

@dataclass
class MatchRecord:
    query_id: str
    gallery_id: str
    verdict: Verdict
    updated_utc: str
    notes: str
    decided_day: str      # YYYY-MM-DD (derived from updated_utc)
    decided_week: str     # YYYY-Www  (ISO week)
    decided_month: str    # YYYY-MM

    # Selected query metadata (prefixed with 'q_')
    q_meta: Row

    # Selected gallery metadata (prefixed with 'g_')
    g_meta: Row


@dataclass
class PastMatchesDataset:
    # One row per decided (query, gallery) pair (latest decision only)
    records: List[MatchRecord]

    # Quick summaries for visualizations
    totals_by_verdict: Dict[Verdict, int]
    timeline_daily: List[Tuple[str, int, int, int, int]]       # (day, yes, maybe, no, total), sorted by day asc
    per_query_counts: Dict[str, Dict[Verdict, int]]            # query_id -> {yes, maybe, no, total}
    per_gallery_counts: Dict[str, Dict[Verdict, int]]          # gallery_id -> {yes, maybe, no, total}


# ------------------------ internals & helpers ------------------------

_ALLOWED_VERDICTS = {"yes", "maybe", "no"}

def _parse_utc_to_parts(iso_utc: str) -> Tuple[str, str, str]:
    """
    Return (day, week, month) where:
      - day   = YYYY-MM-DD
      - week  = YYYY-Www (ISO week)
      - month = YYYY-MM
    Robust to 'Z' suffix and missing/invalid timestamps.
    """
    s = (iso_utc or "").strip()
    if not s:
        return "", "", ""
    try:
        # accept both "...Z" and "...+00:00"
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return "", "", ""
    day = dt.date().isoformat()
    iso_year, iso_week, _ = dt.isocalendar()
    week = f"{iso_year:04d}-W{iso_week:02d}"
    month = f"{dt.year:04d}-{dt.month:02d}"
    return day, week, month


def _label_csv_paths_for_read() -> List[Path]:
    """
    Find all _second_order_labels.csv files under all plausible Queries roots.
    Order: legacy 'querries' first, then 'queries' (so newer rows win on collisions).
    """
    paths: List[Path] = []
    for root in roots_for_read("Queries"):  # handles querries and queries, in stable order
        if not root.exists():
            continue
        for qdir in sorted([p for p in root.iterdir() if p.is_dir()]):
            p = qdir / "_second_order_labels.csv"
            if p.exists():
                paths.append(p)
    return paths  # :contentReference[oaicite:1]{index=1}


def _iter_label_rows_all() -> Iterable[Row]:
    for p in _label_csv_paths_for_read():
        for r in read_rows(p):
            yield r  # BOM/whitespace-tolerant keys handled by read_rows. :contentReference[oaicite:2]{index=2}


def _latest_label_rows_all_pairs() -> List[Row]:
    """
    Read every labels CSV, reduce to the latest row per (query_id, gallery_id).
    Only keep rows with verdict ∈ {"yes","maybe","no"}.
    """
    rows_for_reduce: List[Row] = []
    for r in _iter_label_rows_all():
        qid = normalize_id_value(r.get("query_id", ""))
        gid = normalize_id_value(r.get("gallery_id", ""))
        if not qid or not gid:
            continue
        v = (r.get("verdict", "") or "").strip().lower()
        if v not in _ALLOWED_VERDICTS:
            continue
        # Augment with a composite key so we can reuse last_row_per_id()
        r2 = dict(r)
        r2["query_id"] = qid
        r2["gallery_id"] = gid
        r2["verdict"] = v
        r2["pair_id"] = f"{qid}||{gid}"
        rows_for_reduce.append(r2)

    # "Last row wins" (append-only semantics). We supplied 'pair_id' so the reducer
    # can find the ID column directly. It also ignores pure-ID rows with no payload. :contentReference[oaicite:3]{index=3}
    latest_map = last_row_per_id(rows_for_reduce, "pair_id")
    latest_rows: List[Row] = []
    for _, row in latest_map.items():
        row = dict(row)
        row.pop("pair_id", None)
        latest_rows.append(row)
    return latest_rows  # :contentReference[oaicite:4]{index=4}


def _load_latest_metadata_maps() -> Tuple[Dict[str, Row], Dict[str, Row]]:
    """
    Return (queries_by_id, gallery_by_id) where each value is the latest row per ID
    from all plausible metadata CSVs (new + legacy handled).
    Keys are normalized IDs; each row contains canonical headers.
    """
    # Queries
    q_id = id_column_name("Queries")
    q_rows = read_rows_multi(metadata_csv_paths_for_read("Queries"))
    q_latest = last_row_per_id(q_rows, q_id)  # normalized ID keys. :contentReference[oaicite:5]{index=5}

    # Gallery
    g_id = id_column_name("Gallery")
    g_rows = read_rows_multi(metadata_csv_paths_for_read("Gallery"))
    g_latest = last_row_per_id(g_rows, g_id)  # normalized ID keys. :contentReference[oaicite:6]{index=6}

    # Ensure canonical ID columns are present with the normalized key
    q_by_id: Dict[str, Row] = {}
    for _id, r in q_latest.items():
        r2 = dict(r); r2[q_id] = _id; q_by_id[_id] = r2
    g_by_id: Dict[str, Row] = {}
    for _id, r in g_latest.items():
        r2 = dict(r); r2[g_id] = _id; g_by_id[_id] = r2
    return q_by_id, g_by_id


def _prefix_subset(row: Row, header: List[str], id_col: str, prefix: str) -> Row:
    """
    Keep only the columns in 'header' (except the id_col) and prefix their names.
    Missing keys become empty strings.
    """
    out: Row = {}
    for col in header:
        if col == id_col:
            continue
        out[f"{prefix}{col}"] = row.get(col, "")
    return out


def _master_header() -> List[str]:
    """
    Build the Excel-friendly master CSV header from the canonical metadata headers,
    prefixing query columns with 'q_' and gallery with 'g_'.
    """
    q_id = id_column_name("Queries")
    g_id = id_column_name("Gallery")
    base = ["query_id", "gallery_id", "verdict", "updated_utc", "notes",
            "decided_day", "decided_week", "decided_month"]
    q_cols = [f"q_{c}" for c in QUERIES_HEADER if c != q_id]
    g_cols = [f"g_{c}" for c in GALLERY_HEADER if c != g_id]
    return base + q_cols + g_cols  # :contentReference[oaicite:7]{index=7}


# --------------------------- public API ---------------------------

def build_past_matches_dataset() -> PastMatchesDataset:
    """
    Load and assemble everything the Past Matches tab needs:
      - latest labels per (query, gallery)
      - enriched with metadata (query_*, gallery_*)
      - quick summaries for charts
    """
    q_by_id, g_by_id = _load_latest_metadata_maps()
    id_q = id_column_name("Queries")
    id_g = id_column_name("Gallery")

    records: List[MatchRecord] = []
    for r in _latest_label_rows_all_pairs():
        qid = normalize_id_value(r.get("query_id", ""))
        gid = normalize_id_value(r.get("gallery_id", ""))
        verdict = (r.get("verdict", "") or "").strip().lower()
        updated = (r.get("updated_utc", "") or "").strip()
        notes = r.get("notes", "") or ""

        day, week, month = _parse_utc_to_parts(updated)

        q_row = q_by_id.get(qid, {})
        g_row = g_by_id.get(gid, {})

        q_pref = _prefix_subset(q_row, QUERIES_HEADER, id_q, "q_")
        g_pref = _prefix_subset(g_row, GALLERY_HEADER, id_g, "g_")

        records.append(MatchRecord(
            query_id=qid, gallery_id=gid, verdict=verdict, updated_utc=updated, notes=notes,
            decided_day=day, decided_week=week, decided_month=month,
            q_meta=q_pref, g_meta=g_pref
        ))

    # ---- summaries ----
    totals = Counter(rec.verdict for rec in records)
    # daily timeline (sorted)
    by_day = defaultdict(lambda: Counter())
    for rec in records:
        if rec.decided_day:
            by_day[rec.decided_day][rec.verdict] += 1
    timeline = []
    for day in sorted(by_day.keys()):
        c = by_day[day]
        total = c["yes"] + c["maybe"] + c["no"]
        timeline.append((day, c["yes"], c["maybe"], c["no"], total))

    # per query/gallery counts
    per_q = defaultdict(lambda: Counter())
    per_g = defaultdict(lambda: Counter())
    for rec in records:
        per_q[rec.query_id][rec.verdict] += 1
        per_q[rec.query_id]["total"] += 1
        per_g[rec.gallery_id][rec.verdict] += 1
        per_g[rec.gallery_id]["total"] += 1

    # homogenize dicts
    def _std_counts(m: Dict[str, Counter]) -> Dict[str, Dict[str, int]]:
        out: Dict[str, Dict[str, int]] = {}
        for k, c in m.items():
            out[k] = {"yes": c["yes"], "maybe": c["maybe"], "no": c["no"], "total": c["total"]}
        return out

    ds = PastMatchesDataset(
        records=records,
        totals_by_verdict={"yes": totals["yes"], "maybe": totals["maybe"], "no": totals["no"]},
        timeline_daily=timeline,
        per_query_counts=_std_counts(per_q),
        per_gallery_counts=_std_counts(per_g),
    )
    return ds  # :contentReference[oaicite:8]{index=8}


def export_past_matches_master_csv(path: Optional[Path] = None) -> Path:
    """
    Export the latest (query,gallery) decisions joined with metadata to a single
    Excel-friendly CSV (UTF-8 with BOM). Returns the path.
    """
    ds = build_past_matches_dataset()
    out_path = path or (archive_root() / "reports" / "past_matches_master.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    header = _master_header()
    # Create or validate header; we'll overwrite the file after this check. :contentReference[oaicite:9]{index=9}
    ensure_header(out_path, header)

    # Write fresh (header + rows)
    with out_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        for rec in ds.records:
            base = [rec.query_id, rec.gallery_id, rec.verdict, rec.updated_utc, rec.notes,
                    rec.decided_day, rec.decided_week, rec.decided_month]
            # Order must follow _master_header()
            q_vals = [rec.q_meta.get(f"q_{c}", "") for c in QUERIES_HEADER if c != id_column_name("Queries")]
            g_vals = [rec.g_meta.get(f"g_{c}", "") for c in GALLERY_HEADER if c != id_column_name("Gallery")]
            w.writerow(base + q_vals + g_vals)
    return out_path


def export_past_matches_summaries_csv(
    base_dir: Optional[Path] = None
) -> Tuple[Path, Path, Path]:
    """
    Export three compact tables for quick sharing/QA:
      - summary_by_query.csv    (query_id, yes, maybe, no, total)
      - summary_by_gallery.csv  (gallery_id, yes, maybe, no, total)
      - timeline_daily.csv      (date, yes, maybe, no, total)
    Returns (path_by_query, path_by_gallery, path_timeline).
    """
    ds = build_past_matches_dataset()
    base = base_dir or (archive_root() / "reports")
    base.mkdir(parents=True, exist_ok=True)

    p_q = base / "summary_by_query.csv"
    p_g = base / "summary_by_gallery.csv"
    p_t = base / "timeline_daily.csv"

    with p_q.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["query_id", "yes", "maybe", "no", "total"])
        for qid in sorted(ds.per_query_counts.keys()):
            c = ds.per_query_counts[qid]; w.writerow([qid, c["yes"], c["maybe"], c["no"], c["total"]])

    with p_g.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["gallery_id", "yes", "maybe", "no", "total"])
        for gid in sorted(ds.per_gallery_counts.keys()):
            c = ds.per_gallery_counts[gid]; w.writerow([gid, c["yes"], c["maybe"], c["no"], c["total"]])

    with p_t.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["date", "yes", "maybe", "no", "total"])
        for day, y, m, n, t in ds.timeline_daily:
            w.writerow([day, y, m, n, t])

    return p_q, p_g, p_t
