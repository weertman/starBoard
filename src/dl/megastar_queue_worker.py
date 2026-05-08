from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Callable

from src.dl.megastar_artifact_lock import megastar_artifact_lock
from src.dl.megastar_queue import (
    QueuedJob,
    claim_next_jobs,
    get_queue_status,
    init_queue,
    mark_jobs_completed,
    mark_jobs_failed,
    requeue_running_jobs,
    enqueue_pending_registry_ids,
)
from src.dl.registry import DLRegistry


@dataclass(frozen=True)
class WorkerResult:
    state: str
    message: str
    claimed: int = 0
    completed: int = 0


def _is_still_pending(registry: DLRegistry, job: QueuedJob) -> bool:
    if job.target == "Gallery":
        return job.id_str in registry.pending_ids.gallery
    return job.id_str in registry.pending_ids.queries


def _mark_claimed_jobs_pending(jobs: list[QueuedJob]) -> None:
    registry = DLRegistry.reload()
    for job in jobs:
        registry.add_pending_id(job.target, job.id_str)


def _is_missing_baseline_message(message: str) -> bool:
    lowered = (message or "").lower()
    return "incremental update requires baseline" in lowered or "full precomputation once" in lowered


def run_incremental_pending_once(
    *,
    model_key: str,
    batch_size: int = 1,
    include_verification: bool = False,
) -> tuple[bool, str]:
    from src.dl.precompute import PrecomputeWorker

    result: dict[str, object] = {"ok": False, "message": "Precompute did not finish"}

    def on_finished(ok: bool, message: str) -> None:
        result["ok"] = bool(ok)
        result["message"] = str(message)

    worker = PrecomputeWorker(
        model_key=model_key,
        use_tta=True,
        use_reranking=True,
        batch_size=batch_size,
        include_gallery=True,
        include_queries=True,
        only_pending=True,
        speed_mode="auto",
        include_verification=include_verification,
    )
    worker.finished.connect(on_finished)
    worker._run_precomputation()
    return bool(result["ok"]), str(result["message"])


def run_once(
    *,
    batch_size: int = 1,
    include_verification: bool = False,
    claim_limit: int = 50,
    runner: Callable[..., tuple[bool, str]] = run_incremental_pending_once,
    lock_timeout_seconds: float | None = 1.0,
) -> WorkerResult:
    init_queue()
    registry = DLRegistry.reload()
    model_key = registry.active_model
    if not model_key or model_key not in registry.models or not registry.models[model_key].precomputed:
        return WorkerResult("full_precompute_required", "No active precomputed MegaStar model")

    enqueue_pending_registry_ids(reason="worker_registry_pending_scan")
    worker_id = f"megastar-queue-worker:{os.getpid()}"
    jobs = claim_next_jobs(worker_id, limit=claim_limit, model_key=model_key)
    if not jobs:
        return WorkerResult("idle", "No queued MegaStar jobs")

    job_ids = [job.id for job in jobs]
    try:
        with megastar_artifact_lock(timeout_seconds=lock_timeout_seconds):
            _mark_claimed_jobs_pending(jobs)
            ok, message = runner(
                model_key=model_key,
                batch_size=batch_size,
                include_verification=include_verification,
            )
    except TimeoutError as exc:
        requeue_running_jobs(job_ids)
        return WorkerResult("retry", str(exc), claimed=len(jobs))
    except Exception as exc:
        mark_jobs_failed(job_ids, str(exc), retry=False)
        return WorkerResult("failed", str(exc), claimed=len(jobs))

    if not ok:
        if _is_missing_baseline_message(message):
            requeue_running_jobs(job_ids)
            return WorkerResult("full_precompute_required", message, claimed=len(jobs))
        mark_jobs_failed(job_ids, message, retry=False)
        return WorkerResult("failed", message, claimed=len(jobs))

    registry = DLRegistry.reload()
    completed = [job.id for job in jobs if not _is_still_pending(registry, job)]
    still_pending = [job.id for job in jobs if _is_still_pending(registry, job)]
    mark_jobs_completed(completed)
    requeue_running_jobs(still_pending)
    if still_pending:
        return WorkerResult(
            "pending_remains",
            f"Precompute finished but {len(still_pending)} claimed IDs are still pending",
            claimed=len(jobs),
            completed=len(completed),
        )
    return WorkerResult("completed", message, claimed=len(jobs), completed=len(completed))


def _status_json() -> str:
    return json.dumps(get_queue_status(), indent=2, sort_keys=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run or inspect the starBoard MegaStar background queue worker.")
    parser.add_argument("--once", action="store_true", help="Process at most one coalesced queue pass, then exit.")
    parser.add_argument("--status", action="store_true", help="Print queue status as JSON and exit.")
    parser.add_argument("--poll-seconds", type=float, default=None, help="Run continuously and poll at this interval.")
    parser.add_argument("--batch-size", type=int, default=1, help="Embedding inference batch size. Default: 1.")
    parser.add_argument("--include-verification", action="store_true", help="Also run verification incremental precompute.")
    args = parser.parse_args(argv)

    if args.status:
        print(_status_json())
        return 0

    if args.once or args.poll_seconds is None:
        result = run_once(batch_size=args.batch_size, include_verification=args.include_verification)
        print(json.dumps(result.__dict__, indent=2, sort_keys=True))
        return 0 if result.state not in {"failed"} else 1

    while True:
        result = run_once(batch_size=args.batch_size, include_verification=args.include_verification)
        print(json.dumps(result.__dict__, sort_keys=True), flush=True)
        if result.state == "completed" and get_queue_status().get("queued", 0):
            continue
        time.sleep(max(args.poll_seconds, 1.0))


if __name__ == "__main__":
    raise SystemExit(main())
