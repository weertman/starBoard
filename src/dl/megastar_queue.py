from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Sequence

from src.dl.registry import DLRegistry, DEFAULT_MODEL_KEY

OPEN_STATUSES = ("queued", "running")
VALID_TARGETS = {"Gallery", "Queries"}


@dataclass(frozen=True)
class QueuedJob:
    id: int
    model_key: str
    target: str
    id_str: str
    reason: str
    source: str
    changed_paths_json: str
    status: str
    attempts: int
    error: str
    created_utc: str
    updated_utc: str
    claimed_by: str
    claim_expires_utc: str | None
    completed_utc: str | None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def queue_db_path() -> Path:
    return DLRegistry.get_precompute_root() / "megastar_queue.sqlite"


def _connect() -> sqlite3.Connection:
    path = queue_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_queue() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS megastar_jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_key TEXT NOT NULL,
                target TEXT NOT NULL CHECK(target IN ('Gallery', 'Queries')),
                id_str TEXT NOT NULL,
                reason TEXT NOT NULL DEFAULT '',
                source TEXT NOT NULL DEFAULT '',
                changed_paths_json TEXT NOT NULL DEFAULT '[]',
                status TEXT NOT NULL CHECK(status IN ('queued', 'running', 'completed', 'failed', 'cancelled')),
                attempts INTEGER NOT NULL DEFAULT 0,
                error TEXT NOT NULL DEFAULT '',
                created_utc TEXT NOT NULL,
                updated_utc TEXT NOT NULL,
                claimed_by TEXT NOT NULL DEFAULT '',
                claim_expires_utc TEXT,
                completed_utc TEXT
            )
            """
        )
        conn.execute("DROP INDEX IF EXISTS idx_megastar_open_job")
        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_megastar_queued_job
            ON megastar_jobs(model_key, target, id_str)
            WHERE status = 'queued'
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_megastar_jobs_status_updated
            ON megastar_jobs(status, updated_utc)
            """
        )


def _canonical_target(target: str) -> str:
    if target == "Gallery" or target.lower() == "gallery":
        return "Gallery"
    if target == "Queries" or target.lower() in {"queries", "query", "querries"}:
        return "Queries"
    raise ValueError("target must be 'Gallery' or 'Queries'")


def _normalize_paths(paths: Sequence[Path | str] | None) -> list[str]:
    if not paths:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for path in paths:
        value = str(path)
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _merge_paths(existing_json: str, new_paths: Sequence[Path | str] | None) -> str:
    try:
        existing_raw = json.loads(existing_json or "[]")
    except Exception:
        existing_raw = []
    merged = _normalize_paths([str(p) for p in existing_raw if str(p)])
    seen = set(merged)
    for value in _normalize_paths(new_paths):
        if value not in seen:
            seen.add(value)
            merged.append(value)
    return json.dumps(merged)


def _default_model_key() -> str:
    try:
        registry = DLRegistry.reload()
        if registry.active_model:
            return registry.active_model
    except Exception:
        pass
    return DEFAULT_MODEL_KEY


def _mark_registry_pending(target: str, id_str: str) -> None:
    registry = DLRegistry.reload()
    registry.add_pending_id(target, id_str)


def enqueue_identity_update(
    target: str,
    id_str: str,
    *,
    model_key: str | None = None,
    changed_paths: Sequence[Path | str] | None = None,
    reason: str = "",
    source: str = "",
) -> int | None:
    target = _canonical_target(target)
    id_str = (id_str or "").strip()
    if not id_str:
        return None
    model_key = model_key or _default_model_key()
    init_queue()
    now = _utc_now()
    paths_json = json.dumps(_normalize_paths(changed_paths))
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT * FROM megastar_jobs
            WHERE model_key = ? AND target = ? AND id_str = ? AND status = 'queued'
            ORDER BY id LIMIT 1
            """,
            (model_key, target, id_str),
        ).fetchone()
        if row is not None:
            merged_paths = _merge_paths(row["changed_paths_json"], changed_paths)
            conn.execute(
                """
                UPDATE megastar_jobs
                SET reason = COALESCE(NULLIF(?, ''), reason),
                    source = COALESCE(NULLIF(?, ''), source),
                    changed_paths_json = ?,
                    updated_utc = ?
                WHERE id = ?
                """,
                (reason, source, merged_paths, now, row["id"]),
            )
            job_id = int(row["id"])
        else:
            cur = conn.execute(
                """
                INSERT INTO megastar_jobs (
                    model_key, target, id_str, reason, source, changed_paths_json,
                    status, attempts, error, created_utc, updated_utc
                ) VALUES (?, ?, ?, ?, ?, ?, 'queued', 0, '', ?, ?)
                """,
                (model_key, target, id_str, reason, source, paths_json, now, now),
            )
            job_id = int(cur.lastrowid)
    _mark_registry_pending(target, id_str)
    return job_id


def _row_to_job(row: sqlite3.Row) -> QueuedJob:
    return QueuedJob(
        id=int(row["id"]),
        model_key=str(row["model_key"]),
        target=str(row["target"]),
        id_str=str(row["id_str"]),
        reason=str(row["reason"]),
        source=str(row["source"]),
        changed_paths_json=str(row["changed_paths_json"]),
        status=str(row["status"]),
        attempts=int(row["attempts"]),
        error=str(row["error"]),
        created_utc=str(row["created_utc"]),
        updated_utc=str(row["updated_utc"]),
        claimed_by=str(row["claimed_by"]),
        claim_expires_utc=row["claim_expires_utc"],
        completed_utc=row["completed_utc"],
    )


def list_open_jobs(model_key: str | None = None) -> list[QueuedJob]:
    init_queue()
    sql = "SELECT * FROM megastar_jobs WHERE status IN ('queued', 'running')"
    params: tuple[str, ...] = ()
    if model_key is not None:
        sql += " AND model_key = ?"
        params = (model_key,)
    sql += " ORDER BY id"
    with _connect() as conn:
        return [_row_to_job(r) for r in conn.execute(sql, params).fetchall()]


def get_queue_status(model_key: str | None = None) -> dict:
    init_queue()
    counts = {"queued": 0, "running": 0, "completed": 0, "failed": 0, "cancelled": 0}
    sql = "SELECT status, COUNT(*) AS n FROM megastar_jobs"
    params: tuple[str, ...] = ()
    if model_key is not None:
        sql += " WHERE model_key = ?"
        params = (model_key,)
    sql += " GROUP BY status"
    with _connect() as conn:
        for row in conn.execute(sql, params):
            counts[str(row["status"])] = int(row["n"])
        last = conn.execute(
            "SELECT error FROM megastar_jobs WHERE error != '' ORDER BY updated_utc DESC, id DESC LIMIT 1"
        ).fetchone()
    counts["open"] = counts["queued"] + counts["running"]
    counts["last_error"] = str(last["error"]) if last else ""
    try:
        registry = DLRegistry.reload()
        active = registry.active_model
        if not active or active not in registry.models or not registry.models[active].precomputed:
            counts["worker_state"] = "full_precompute_required" if counts["open"] else "idle"
        elif counts["running"]:
            counts["worker_state"] = "running"
        elif counts["queued"]:
            counts["worker_state"] = "queued"
        else:
            counts["worker_state"] = "idle"
        counts["active_model"] = active
    except Exception:
        counts["worker_state"] = "unknown"
        counts["active_model"] = None
    counts["db_path"] = str(queue_db_path())
    return counts


def claim_next_jobs(worker_id: str, *, limit: int = 50, lease_seconds: int = 3600, model_key: str | None = None) -> list[QueuedJob]:
    init_queue()
    now = _utc_now()
    claim_expires = (datetime.now(timezone.utc) + timedelta(seconds=lease_seconds)).isoformat()
    with _connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            """
            UPDATE megastar_jobs
            SET status = 'cancelled', error = 'superseded_by_queued_followup', claimed_by = '',
                claim_expires_utc = NULL, updated_utc = ?
            WHERE status = 'running'
              AND claim_expires_utc IS NOT NULL
              AND claim_expires_utc < ?
              AND EXISTS (
                SELECT 1 FROM megastar_jobs q
                WHERE q.status = 'queued'
                  AND q.model_key = megastar_jobs.model_key
                  AND q.target = megastar_jobs.target
                  AND q.id_str = megastar_jobs.id_str
              )
            """,
            (now, now),
        )
        conn.execute(
            """
            UPDATE megastar_jobs
            SET status = 'queued', claimed_by = '', claim_expires_utc = NULL, updated_utc = ?
            WHERE status = 'running'
              AND claim_expires_utc IS NOT NULL
              AND claim_expires_utc < ?
              AND NOT EXISTS (
                SELECT 1 FROM megastar_jobs q
                WHERE q.status = 'queued'
                  AND q.model_key = megastar_jobs.model_key
                  AND q.target = megastar_jobs.target
                  AND q.id_str = megastar_jobs.id_str
              )
            """,
            (now, now),
        )
        sql = "SELECT * FROM megastar_jobs WHERE status = 'queued'"
        params: list[object] = []
        if model_key is not None:
            sql += " AND model_key = ?"
            params.append(model_key)
        sql += " ORDER BY id LIMIT ?"
        params.append(limit)
        rows = conn.execute(sql, tuple(params)).fetchall()
        ids = [int(r["id"]) for r in rows]
        if ids:
            placeholders = ",".join("?" for _ in ids)
            conn.execute(
                f"""
                UPDATE megastar_jobs
                SET status = 'running', attempts = attempts + 1, claimed_by = ?,
                    claim_expires_utc = ?, updated_utc = ?
                WHERE status = 'queued' AND id IN ({placeholders})
                """,
                (worker_id, claim_expires, now, *ids),
            )
            rows = conn.execute(
                f"SELECT * FROM megastar_jobs WHERE id IN ({placeholders}) ORDER BY id",
                tuple(ids),
            ).fetchall()
        conn.commit()
    return [_row_to_job(r) for r in rows]


def mark_jobs_completed(job_ids: Iterable[int]) -> None:
    ids = [int(i) for i in job_ids]
    if not ids:
        return
    init_queue()
    now = _utc_now()
    placeholders = ",".join("?" for _ in ids)
    with _connect() as conn:
        conn.execute(
            f"""
            UPDATE megastar_jobs
            SET status = 'completed', error = '', claimed_by = '', claim_expires_utc = NULL,
                completed_utc = ?, updated_utc = ?
            WHERE id IN ({placeholders})
            """,
            (now, now, *ids),
        )


def mark_jobs_failed(job_ids: Iterable[int], error: str, *, retry: bool = False) -> None:
    ids = [int(i) for i in job_ids]
    if not ids:
        return
    init_queue()
    now = _utc_now()
    status = "queued" if retry else "failed"
    placeholders = ",".join("?" for _ in ids)
    with _connect() as conn:
        conn.execute(
            f"""
            UPDATE megastar_jobs
            SET status = ?, error = ?, claimed_by = '', claim_expires_utc = NULL, updated_utc = ?
            WHERE id IN ({placeholders})
            """,
            (status, str(error), now, *ids),
        )


def requeue_running_jobs(job_ids: Iterable[int]) -> None:
    ids = [int(i) for i in job_ids]
    if not ids:
        return
    init_queue()
    now = _utc_now()
    placeholders = ",".join("?" for _ in ids)
    with _connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            f"""
            UPDATE megastar_jobs
            SET status = 'cancelled', error = 'superseded_by_queued_followup', claimed_by = '',
                claim_expires_utc = NULL, updated_utc = ?
            WHERE id IN ({placeholders})
              AND status = 'running'
              AND EXISTS (
                SELECT 1 FROM megastar_jobs q
                WHERE q.status = 'queued'
                  AND q.model_key = megastar_jobs.model_key
                  AND q.target = megastar_jobs.target
                  AND q.id_str = megastar_jobs.id_str
              )
            """,
            (now, *ids),
        )
        conn.execute(
            f"""
            UPDATE megastar_jobs
            SET status = 'queued', claimed_by = '', claim_expires_utc = NULL, updated_utc = ?
            WHERE id IN ({placeholders})
              AND status = 'running'
              AND NOT EXISTS (
                SELECT 1 FROM megastar_jobs q
                WHERE q.status = 'queued'
                  AND q.model_key = megastar_jobs.model_key
                  AND q.target = megastar_jobs.target
                  AND q.id_str = megastar_jobs.id_str
              )
            """,
            (now, *ids),
        )
        conn.commit()


def enqueue_pending_registry_ids(*, reason: str = "registry_pending_scan") -> int:
    registry = DLRegistry.reload()
    count = 0
    for gid in registry.pending_ids.gallery:
        if enqueue_identity_update("Gallery", gid, model_key=registry.active_model or DEFAULT_MODEL_KEY, reason=reason):
            count += 1
    for qid in registry.pending_ids.queries:
        if enqueue_identity_update("Queries", qid, model_key=registry.active_model or DEFAULT_MODEL_KEY, reason=reason):
            count += 1
    return count
