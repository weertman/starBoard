# src/data/silence.py
from __future__ import annotations

"""
Utilities for marking Query folders as "silent" so they are hidden in
interactive tabs (First-order / Second-order) without deleting anything.

A Query becomes silent when a small JSON sidecar file is present:

    <archive>/queries/<query_id>/_SILENT.flag

Schema (JSON):
{
  "schema": "starBoard.silence/v1",
  "query_id": "<id>",
  "reason": "merged-yes" | "manual" | "...",
  "notes": "<optional free text>",
  "by": "<optional operator>",
  "effective_tabs": ["first", "second"],  # future-proofing; currently informational
  "updated_utc": "YYYY-MM-DDTHH:MM:SSZ"
}

This file may be removed to un-silence the Query.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import logging

from .archive_paths import queries_root, roots_for_read
from .csv_io import normalize_id_value

log = logging.getLogger("starBoard.data.silence")

_FLAG_NAME = "_SILENT.flag"
_SCHEMA = "starBoard.silence/v1"


@dataclass
class SilenceInfo:
    query_id: str
    reason: str
    notes: str
    by: str
    effective_tabs: List[str]
    updated_utc: str
    path: Path

    @staticmethod
    def empty(query_id: str, path: Path) -> "SilenceInfo":
        return SilenceInfo(
            query_id=query_id,
            reason="",
            notes="",
            by="",
            effective_tabs=[],
            updated_utc="",
            path=path,
        )


# ---------------------------- paths ----------------------------

def flag_path_for_query(query_id: str, *, prefer_new: bool = True) -> Path:
    """
    Preferred location to WRITE a silence flag for a query.
    Uses the current 'queries' root (not legacy 'querries').
    """
    qid = normalize_id_value(query_id)
    return queries_root(prefer_new=prefer_new) / qid / _FLAG_NAME


def all_flag_paths_for_query(query_id: str) -> List[Path]:
    """
    All plausible locations to READ a silence flag for a query,
    across both legacy and new query roots.
    """
    qid = normalize_id_value(query_id)
    paths: List[Path] = []
    for root in roots_for_read("Queries"):
        paths.append(root / qid / _FLAG_NAME)
    # de-dup while preserving order
    out: List[Path] = []
    seen = set()
    for p in paths:
        k = str(p.resolve()) if p.exists() else str(p)
        if k not in seen:
            out.append(p)
            seen.add(k)
    return out


# ---------------------------- core API ----------------------------

def is_silent_query(query_id: str) -> bool:
    """
    Return True if the Query ID is marked 'silent' either by the modern
    batch-based marker (<query>/_starboard_silent.json with non-empty 'batches')
    OR by the legacy/manual flag file (<query>/_SILENT.flag).

    Robust to malformed JSON: treat as silent (fail-closed).
    """
    qid = normalize_id_value(query_id)

    # 1) Modern batch-based marker (_starboard_silent.json) across all plausible roots
    STARBOARD_MARKER = "_starboard_silent.json"
    try:
        for root in roots_for_read("Queries"):
            p = root / qid / STARBOARD_MARKER
            if p.exists():
                try:
                    obj = json.loads(p.read_text(encoding="utf-8"))
                    batches = obj.get("batches") or []
                    if len(batches) > 0:
                        return True
                except Exception:
                    # Malformed marker -> consider it silent to be safe
                    return True
    except Exception:
        # If enumeration fails, proceed to legacy flag check
        pass

    # 2) Legacy/manual flag (_SILENT.flag)
    for p in all_flag_paths_for_query(qid):
        if p.exists():
            return True

    return False



def load_silence_info(query_id: str) -> SilenceInfo:
    """
    Best-effort parse of the silence flag JSON. If multiple plausible
    paths exist, the first existing flag encountered is returned.
    Returns an empty info object if no flag exists or it cannot be read.
    """
    for p in all_flag_paths_for_query(query_id):
        if not p.exists():
            continue
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
            if not isinstance(doc, dict):
                continue
            return SilenceInfo(
                query_id=normalize_id_value(doc.get("query_id", "") or query_id),
                reason=str(doc.get("reason", "") or ""),
                notes=str(doc.get("notes", "") or ""),
                by=str(doc.get("by", "") or ""),
                effective_tabs=list(doc.get("effective_tabs", []) or []),
                updated_utc=str(doc.get("updated_utc", "") or ""),
                path=p,
            )
        except Exception as e:
            log.warning("silence.load failed for %s: %s", p, e)
            return SilenceInfo.empty(query_id, p)
    # no flag anywhere
    return SilenceInfo.empty(query_id, flag_path_for_query(query_id))


def set_silent_query(
    query_id: str,
    *,
    reason: str = "merged-yes",
    by: str = "",
    notes: str = "",
    effective_tabs: Optional[List[str]] = None,
) -> Path:
    """
    Create or overwrite the silence flag for the given query.
    Returns the path to the flag file.
    """
    qid = normalize_id_value(query_id)
    path = flag_path_for_query(qid, prefer_new=True)
    path.parent.mkdir(parents=True, exist_ok=True)

    obj: Dict[str, object] = {
        "schema": _SCHEMA,
        "query_id": qid,
        "reason": (reason or "").strip(),
        "notes": notes or "",
        "by": by or "",
        "effective_tabs": list(effective_tabs or ["first", "second"]),
        "updated_utc": datetime.utcnow().isoformat() + "Z",
    }
    try:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
        tmp.replace(path)
        log.info("silence.set query_id=%s path=%s reason=%s", qid, str(path), reason)
    except Exception as e:
        log.error("silence.set failed query_id=%s: %s", qid, e)
        raise
    return path


def clear_silent_query(query_id: str) -> bool:
    """
    Remove the silence flag from all plausible locations.
    Returns True if at least one was removed.
    """
    any_removed = False
    for p in all_flag_paths_for_query(query_id):
        try:
            if p.exists():
                p.unlink()
                any_removed = True
                log.info("silence.clear removed %s", str(p))
        except Exception as e:
            log.warning("silence.clear failed for %s: %s", p, e)
    return any_removed


def list_silent_queries() -> List[str]:
    """
    Scan all plausible Query roots and return IDs that currently have a
    `_SILENT.flag` file present. Sorted alphabetically.
    """
    ids: set[str] = set()
    for root in roots_for_read("Queries"):
        if not root.exists():
            continue
        for child in root.iterdir():
            if child.is_dir():
                flag = child / _FLAG_NAME
                if flag.exists():
                    ids.add(child.name)
    return sorted(ids)
