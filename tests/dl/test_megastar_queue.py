from __future__ import annotations

import json
from pathlib import Path

from src.dl.megastar_queue import (
    enqueue_identity_update,
    get_queue_status,
    list_open_jobs,
    mark_jobs_completed,
    queue_db_path,
)
from src.dl.registry import DLRegistry, DEFAULT_MODEL_KEY


def test_enqueue_identity_update_creates_durable_open_job_and_marks_registry_pending(tmp_path, monkeypatch):
    monkeypatch.setenv("STARBOARD_ARCHIVE_DIR", str(tmp_path / "archive"))

    job_id = enqueue_identity_update(
        "Gallery",
        "g-001",
        changed_paths=[tmp_path / "archive" / "gallery" / "g-001" / "04_29_26" / "a.jpg"],
        reason="test",
        source="pytest",
    )

    assert job_id is not None
    assert queue_db_path().exists()

    status = get_queue_status(DEFAULT_MODEL_KEY)
    assert status["queued"] == 1
    assert status["running"] == 0
    assert status["failed"] == 0

    jobs = list_open_jobs(DEFAULT_MODEL_KEY)
    assert len(jobs) == 1
    assert jobs[0].target == "Gallery"
    assert jobs[0].id_str == "g-001"
    assert json.loads(jobs[0].changed_paths_json)[0].endswith("a.jpg")

    registry = DLRegistry.reload()
    assert registry.pending_ids.gallery == ["g-001"]
    assert registry.pending_ids.queries == []


def test_duplicate_enqueue_coalesces_open_job_and_merges_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("STARBOARD_ARCHIVE_DIR", str(tmp_path / "archive"))

    first = enqueue_identity_update("Queries", "q-001", changed_paths=["one.jpg"], reason="first")
    second = enqueue_identity_update("Queries", "q-001", changed_paths=["two.jpg"], reason="second")

    assert first == second
    jobs = list_open_jobs(DEFAULT_MODEL_KEY)
    assert len(jobs) == 1
    paths = json.loads(jobs[0].changed_paths_json)
    assert paths == ["one.jpg", "two.jpg"]

    registry = DLRegistry.reload()
    assert registry.pending_ids.queries == ["q-001"]


def test_enqueue_while_job_running_creates_followup_queued_job(tmp_path, monkeypatch):
    monkeypatch.setenv("STARBOARD_ARCHIVE_DIR", str(tmp_path / "archive"))

    first = enqueue_identity_update("Gallery", "g-001", changed_paths=["first.jpg"])
    from src.dl.megastar_queue import claim_next_jobs
    claimed = claim_next_jobs("worker", limit=1)
    assert [job.id for job in claimed] == [first]

    second = enqueue_identity_update("Gallery", "g-001", changed_paths=["second.jpg"])

    assert second != first
    jobs = list_open_jobs(DEFAULT_MODEL_KEY)
    assert [job.status for job in jobs] == ["running", "queued"]


def test_requeue_running_job_with_followup_queued_job_does_not_violate_unique_index(tmp_path, monkeypatch):
    monkeypatch.setenv("STARBOARD_ARCHIVE_DIR", str(tmp_path / "archive"))

    first = enqueue_identity_update("Gallery", "g-001", changed_paths=["first.jpg"])
    from src.dl.megastar_queue import claim_next_jobs, get_queue_status, requeue_running_jobs
    claim_next_jobs("worker", limit=1)
    second = enqueue_identity_update("Gallery", "g-001", changed_paths=["second.jpg"])

    requeue_running_jobs([first])

    jobs = list_open_jobs(DEFAULT_MODEL_KEY)
    assert [job.id for job in jobs] == [second]
    assert jobs[0].status == "queued"
    assert get_queue_status(DEFAULT_MODEL_KEY)["cancelled"] == 1


def test_completed_job_allows_future_enqueue_for_same_identity(tmp_path, monkeypatch):
    monkeypatch.setenv("STARBOARD_ARCHIVE_DIR", str(tmp_path / "archive"))

    first = enqueue_identity_update("Gallery", "g-001")
    mark_jobs_completed([first])
    second = enqueue_identity_update("Gallery", "g-001")

    assert second != first
    status = get_queue_status(DEFAULT_MODEL_KEY)
    assert status["completed"] == 1
    assert status["queued"] == 1


def test_invalid_target_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("STARBOARD_ARCHIVE_DIR", str(tmp_path / "archive"))

    try:
        enqueue_identity_update("Bad", "x")
    except ValueError as exc:
        assert "target" in str(exc)
    else:
        raise AssertionError("expected ValueError")
