# src/data/merge_yes.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Iterable, Optional, Set
from datetime import datetime
import shutil, uuid, json, csv, logging

from .archive_paths import gallery_root, queries_root
from .id_registry import list_ids
from .compare_labels import load_latest_map_for_query
from .validators import validate_mmddyy_string

log = logging.getLogger("starBoard.data.merge_yes")

SILENT_MARKER_FILENAME = "_starboard_silent.json"
HISTORY_CSV_NAME = "_merge_history.csv"

_HIST_HEADER = [
    "batch_id", "timestamp_utc", "operation", "gallery_id", "query_id",
    "src_rel", "dest_rel", "kind", "status", "note"
]

@dataclass
class MergePlanItem:
    query_id: str
    src_dir: Path          # absolute path (encounter dir in query)
    dest_dir: Path         # absolute path (encounter dir to be created under gallery)

@dataclass
class MergeReport:
    batch_id: str
    gallery_id: str
    num_queries: int
    num_encounter_dirs: int
    created_dirs: List[Path]
    errors: List[str]
    dry_run: bool = False

def _history_path(gallery_id: str) -> Path:
    return gallery_root() / gallery_id / HISTORY_CSV_NAME

def _ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)

def _ensure_unique_dir(dest: Path) -> Path:
    """Return a unique directory path by suffixing ' (n)' if needed."""
    if not dest.exists():
        return dest
    base = dest.name
    parent = dest.parent
    n = 1
    while True:
        cand = parent / f"{base} ({n})"
        if not cand.exists():
            return cand
        n += 1

def _list_encounter_dirs(query_id: str) -> List[Path]:
    """Immediate child directories of a query folder that look like MM_DD_YY*."""
    qroot = queries_root(prefer_new=True) / query_id
    if not qroot.exists():
        return []
    out: List[Path] = []
    for child in sorted(p for p in qroot.iterdir() if p.is_dir()):
        name = child.name
        if validate_mmddyy_string(name).ok:
            out.append(child)
    return out

def _read_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

def _append_rows(path: Path, rows: Iterable[Dict[str, str]]) -> None:
    _ensure_parent(path)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=_HIST_HEADER)
        if not exists:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in _HIST_HEADER})

def _silent_path(query_id: str) -> Path:
    return queries_root(prefer_new=True) / query_id / SILENT_MARKER_FILENAME

def is_query_silent(query_id: str) -> bool:
    """True if the query has any active silent batches."""
    p = _silent_path(query_id)
    if not p.exists():
        return False
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        batches = obj.get("batches") or []
        return len(batches) > 0
    except Exception:
        # On malformed file, treat as silent to be safe
        return True

def _add_silent_batch(query_id: str, batch_id: str) -> None:
    p = _silent_path(query_id)
    batches: List[str] = []
    if p.exists():
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            batches = list(dict.fromkeys([str(x) for x in (obj.get("batches") or [])]))
        except Exception:
            batches = []
    if batch_id not in batches:
        batches.append(batch_id)
    _ensure_parent(p)
    p.write_text(json.dumps({"batches": batches}, indent=2), encoding="utf-8")

def _remove_silent_batch(query_id: str, batch_id: str) -> None:
    p = _silent_path(query_id)
    if not p.exists():
        return
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        batches = [str(x) for x in (obj.get("batches") or []) if str(x) != batch_id]
        if batches:
            p.write_text(json.dumps({"batches": batches}, indent=2), encoding="utf-8")
        else:
            p.unlink(missing_ok=True)
    except Exception:
        # If anything is wrong, best effort: delete file (unsilence)
        try:
            p.unlink(missing_ok=True)
        except Exception:
            pass

# ---------- discovery helpers (existing + new) ----------

def list_yes_queries_for_gallery(gallery_id: str) -> List[str]:
    """All query_ids that currently have verdict=='yes' against gallery_id."""
    qs = list_ids("Queries")
    out: List[str] = []
    for qid in qs:
        try:
            latest = load_latest_map_for_query(qid)
            row = latest.get(gallery_id)
            v = (row or {}).get("verdict", "") or ""
            if v.strip().lower() == "yes":
                out.append(qid)
        except Exception:
            continue
    return sorted(out)

def list_mergeable_queries_for_gallery(gallery_id: str, *, require_encounters: bool = True) -> List[str]:
    """
    YES queries for *gallery_id* that are **not silent** and (optionally) have ≥1 encounter directory.
    This is what drives the Merge UI's gallery filtering.
    """
    qids = [qid for qid in list_yes_queries_for_gallery(gallery_id) if not is_query_silent(qid)]
    if not require_encounters:
        return qids
    out: List[str] = []
    for qid in qids:
        if _list_encounter_dirs(qid):
            out.append(qid)
    return out

def list_galleries_with_merge_candidates(*, require_encounters: bool = True) -> List[str]:
    """
    Gallery IDs that currently have at least one merge‑able YES query.
    After a merge, queries are silenced; those galleries drop out automatically.
    """
    out: List[str] = []
    for gid in list_ids("Gallery"):
        if list_mergeable_queries_for_gallery(gid, require_encounters=require_encounters):
            out.append(gid)
    return sorted(out)

def list_galleries_with_history() -> List[str]:
    """All galleries that have at least one merge history row."""
    out: List[str] = []
    for gid in list_ids("Gallery"):
        rows = _read_rows(_history_path(gid))
        if any(r.get("operation") == "merge" for r in rows):
            out.append(gid)
    return sorted(out)

def list_batches_for_gallery(gallery_id: str) -> List[Dict[str, str]]:
    """
    Return a list of batches (most recent last) with summary fields:
    {batch_id, timestamp_utc, num_dirs, num_queries}.
    """
    rows = _read_rows(_history_path(gallery_id))
    by_batch: Dict[str, Dict[str, str]] = {}
    # Keep in file order (append-only), so we can preserve chronological order.
    for r in rows:
        if r.get("operation") != "merge":
            continue
        bid = r.get("batch_id") or ""
        if not bid:
            continue
        entry = by_batch.setdefault(bid, {"batch_id": bid, "timestamp_utc": r.get("timestamp_utc", "") or "",
                                          "num_dirs": "0", "num_queries": "0"})
        # Count created directories
        if r.get("kind") == "dir" and r.get("status") == "created":
            entry["num_dirs"] = str(int(entry["num_dirs"]) + 1)
        # Count silenced queries
        if r.get("kind") == "silence_marker" and r.get("status") == "created":
            entry["num_queries"] = str(int(entry["num_queries"]) + 1)
    # Preserve chronological order as in file
    ordered = [by_batch[bid] for bid in [r.get("batch_id","") for r in rows if r.get("operation")=="merge"] if bid in by_batch]
    # Deduplicate while preserving order
    seen: Set[str] = set()
    result: List[Dict[str, str]] = []
    for e in ordered:
        if e["batch_id"] in seen:
            continue
        seen.add(e["batch_id"])
        result.append(e)
    return result

# ---------- planning & execution ----------

def _build_plan(gallery_id: str, query_ids: List[str]) -> List[MergePlanItem]:
    """Plan copying of encounter dirs (prefix MM_DD_YY*) from each query to gallery."""
    dest_base = gallery_root() / gallery_id
    plan: List[MergePlanItem] = []
    for qid in query_ids:
        for enc in _list_encounter_dirs(qid):
            dest_dir = _ensure_unique_dir(dest_base / enc.name)
            plan.append(MergePlanItem(query_id=qid, src_dir=enc, dest_dir=dest_dir))
    return plan

def merge_yeses_for_gallery(gallery_id: str, *, dry_run: bool = False) -> MergeReport:
    """Copy all encounter folders for YES queries into the gallery, with history + silence markers."""
    qids = list_mergeable_queries_for_gallery(gallery_id, require_encounters=True)
    batch_id = datetime.utcnow().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    plan = _build_plan(gallery_id, qids)
    created: List[Path] = []
    errors: List[str] = []
    hist_rows: List[Dict[str, str]] = []
    now = datetime.utcnow().isoformat() + "Z"

    # Perform copy
    if not dry_run:
        try:
            (gallery_root() / gallery_id).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            errors.append(f"Failed to ensure gallery folder: {e}")

    for item in plan:
        src_rel = str(item.src_dir.relative_to(queries_root(prefer_new=True)))
        dest_rel = str(item.dest_dir.relative_to(gallery_root()))
        if dry_run:
            continue
        try:
            shutil.copytree(item.src_dir, item.dest_dir)
            created.append(item.dest_dir)
            hist_rows.append({
                "batch_id": batch_id, "timestamp_utc": now, "operation": "merge",
                "gallery_id": gallery_id, "query_id": item.query_id,
                "src_rel": src_rel, "dest_rel": dest_rel, "kind": "dir",
                "status": "created", "note": ""
            })
        except Exception as e:
            errors.append(f"copytree failed: {item.src_dir} -> {item.dest_dir}: {e}")
            hist_rows.append({
                "batch_id": batch_id, "timestamp_utc": now, "operation": "merge",
                "gallery_id": gallery_id, "query_id": item.query_id,
                "src_rel": src_rel, "dest_rel": dest_rel, "kind": "dir",
                "status": "error", "note": str(e)
            })

    # Silence markers (per query)
    if not dry_run:
        for qid in sorted(set(qids)):
            try:
                _add_silent_batch(qid, batch_id)
                hist_rows.append({
                    "batch_id": batch_id, "timestamp_utc": now, "operation": "merge",
                    "gallery_id": gallery_id, "query_id": qid,
                    "src_rel": f"{qid}/{SILENT_MARKER_FILENAME}",
                    "dest_rel": "", "kind": "silence_marker",
                    "status": "created", "note": ""
                })
            except Exception as e:
                errors.append(f"failed to add silent marker for {qid}: {e}")

    if not dry_run:
        _append_rows(_history_path(gallery_id), hist_rows)

    return MergeReport(
        batch_id=batch_id,
        gallery_id=gallery_id,
        num_queries=len(qids),
        num_encounter_dirs=len(plan),
        created_dirs=created,
        errors=errors,
        dry_run=dry_run
    )

def _last_merge_batch_id(rows: List[Dict[str, str]]) -> Optional[str]:
    # last row with operation='merge' gives the most recent batch
    merges = [r for r in rows if (r.get("operation") == "merge")]
    if not merges:
        return None
    return merges[-1].get("batch_id") or None

def revert_merge_batch_for_gallery(gallery_id: str, batch_id: str) -> MergeReport:
    """Revert the specified merge batch for this gallery."""
    rows = _read_rows(_history_path(gallery_id))
    if not batch_id:
        return MergeReport(batch_id="", gallery_id=gallery_id, num_queries=0,
                           num_encounter_dirs=0, created_dirs=[], errors=["Empty batch id."], dry_run=False)

    # Determine all dirs created and queries silenced in that batch
    created_dirs: List[Path] = []
    affected_queries: Set[str] = set()
    for r in rows:
        if r.get("batch_id") != batch_id:
            continue
        if r.get("operation") != "merge":
            continue
        if r.get("kind") == "dir" and r.get("status") == "created":
            dest_rel = r.get("dest_rel") or ""
            if dest_rel:
                created_dirs.append(gallery_root() / dest_rel)
        if r.get("kind") == "silence_marker":
            qid = r.get("query_id") or ""
            if qid:
                affected_queries.add(qid)

    now = datetime.utcnow().isoformat() + "Z"
    hist_rows: List[Dict[str, str]] = []
    errors: List[str] = []
    removed_count = 0

    # Remove created dirs (best-effort)
    for d in sorted(created_dirs, key=lambda p: len(str(p)), reverse=True):
        try:
            d_abs = d.resolve()
            g_abs = (gallery_root() / gallery_id).resolve()
            if g_abs not in d_abs.parents and d_abs != g_abs and d_abs.parent != g_abs and g_abs / d.name != d_abs:
                status = "skipped"  # safety
            else:
                if d.exists():
                    shutil.rmtree(d)
                    status = "removed"; removed_count += 1
                else:
                    status = "missing"
            hist_rows.append({
                "batch_id": batch_id, "timestamp_utc": now, "operation": "revert",
                "gallery_id": gallery_id, "query_id": "", "src_rel": "", "dest_rel": str(d.relative_to(gallery_root())),
                "kind": "dir", "status": status, "note": ""
            })
        except Exception as e:
            errors.append(f"failed to remove {d}: {e}")
            hist_rows.append({
                "batch_id": batch_id, "timestamp_utc": now, "operation": "revert",
                "gallery_id": gallery_id, "query_id": "", "src_rel": "", "dest_rel": str(d),
                "kind": "dir", "status": "error", "note": str(e)
            })

    # Remove silent markers for this batch (or delete file if empty)
    for qid in sorted(affected_queries):
        try:
            _remove_silent_batch(qid, batch_id)
            hist_rows.append({
                "batch_id": batch_id, "timestamp_utc": now, "operation": "revert",
                "gallery_id": gallery_id, "query_id": qid,
                "src_rel": f"{qid}/{SILENT_MARKER_FILENAME}", "dest_rel": "",
                "kind": "silence_marker", "status": "removed", "note": ""
            })
        except Exception as e:
            errors.append(f"failed to remove silent marker for {qid}: {e}")

    _append_rows(_history_path(gallery_id), hist_rows)

    return MergeReport(
        batch_id=batch_id,
        gallery_id=gallery_id,
        num_queries=len(affected_queries),
        num_encounter_dirs=removed_count,
        created_dirs=[],
        errors=errors,
        dry_run=False
    )

def revert_last_merge_for_gallery(gallery_id: str) -> MergeReport:
    """Revert the most recent merge batch for this gallery."""
    rows = _read_rows(_history_path(gallery_id))
    batch_id = _last_merge_batch_id(rows)
    if not batch_id:
        return MergeReport(batch_id="", gallery_id=gallery_id, num_queries=0,
                           num_encounter_dirs=0, created_dirs=[], errors=["No merge batches found."], dry_run=False)
    return revert_merge_batch_for_gallery(gallery_id, batch_id)

def read_merge_history_for_gallery(gallery_id: str) -> List[Dict[str, str]]:
    """Read history rows for this gallery (empty if none)."""
    return _read_rows(_history_path(gallery_id))
