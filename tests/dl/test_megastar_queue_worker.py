from __future__ import annotations

import sqlite3
from pathlib import Path

from src.dl.megastar_queue import enqueue_identity_update, get_queue_status
from src.dl.megastar_queue_worker import run_once
from src.dl.registry import DLRegistry, ModelEntry, DEFAULT_MODEL_KEY


def _write_active_precomputed_registry(tmp_path, model_key=DEFAULT_MODEL_KEY):
    registry = DLRegistry.reload()
    registry.models[model_key] = ModelEntry(
        checkpoint_path=str(tmp_path / "fake.pth"),
        checkpoint_hash="fake",
        display_name="Fake",
        precomputed=True,
    )
    registry.active_model = model_key
    registry.save()


def test_run_once_reports_full_precompute_required_without_active_precomputed_model(tmp_path, monkeypatch):
    monkeypatch.setenv("STARBOARD_ARCHIVE_DIR", str(tmp_path / "archive"))
    enqueue_identity_update("Gallery", "g-001")

    result = run_once(batch_size=1, runner=lambda **kwargs: (True, "should not run"))

    assert result.state == "full_precompute_required"
    assert get_queue_status()["queued"] == 1


def test_run_once_processes_queue_with_batch_size_one_and_completes_cleared_jobs(tmp_path, monkeypatch):
    monkeypatch.setenv("STARBOARD_ARCHIVE_DIR", str(tmp_path / "archive"))
    _write_active_precomputed_registry(tmp_path)
    enqueue_identity_update("Gallery", "g-001")
    calls = []

    def fake_runner(**kwargs):
        calls.append(kwargs)
        registry = DLRegistry.reload()
        registry.pending_ids.clear()
        registry.save()
        return True, "ok"

    result = run_once(batch_size=1, runner=fake_runner)

    assert result.state == "completed"
    assert calls[0]["batch_size"] == 1
    assert calls[0]["model_key"] == DEFAULT_MODEL_KEY
    status = get_queue_status()
    assert status["queued"] == 0
    assert status["completed"] == 1


def test_run_once_keeps_pending_job_queued_if_runner_succeeds_but_registry_still_pending(tmp_path, monkeypatch):
    monkeypatch.setenv("STARBOARD_ARCHIVE_DIR", str(tmp_path / "archive"))
    _write_active_precomputed_registry(tmp_path)
    enqueue_identity_update("Queries", "q-001")

    result = run_once(batch_size=1, runner=lambda **kwargs: (True, "no-op"))

    assert result.state == "pending_remains"
    status = get_queue_status()
    assert status["queued"] == 1
    assert status["completed"] == 0


def test_run_once_marks_failed_when_runner_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("STARBOARD_ARCHIVE_DIR", str(tmp_path / "archive"))
    _write_active_precomputed_registry(tmp_path)
    enqueue_identity_update("Gallery", "g-001")

    result = run_once(batch_size=1, runner=lambda **kwargs: (False, "boom"))

    assert result.state == "failed"
    status = get_queue_status()
    assert status["queued"] == 0
    assert status["failed"] == 1


def test_run_once_consumes_registry_only_pending_ids(tmp_path, monkeypatch):
    monkeypatch.setenv("STARBOARD_ARCHIVE_DIR", str(tmp_path / "archive"))
    _write_active_precomputed_registry(tmp_path)
    registry = DLRegistry.reload()
    registry.add_pending_id("Gallery", "legacy-pending")

    def fake_runner(**kwargs):
        registry = DLRegistry.reload()
        registry.pending_ids.clear()
        registry.save()
        return True, "ok"

    result = run_once(batch_size=1, runner=fake_runner)

    assert result.state == "completed"
    assert get_queue_status()["completed"] == 1


def test_run_once_uses_active_model_for_new_queue_work(tmp_path, monkeypatch):
    monkeypatch.setenv("STARBOARD_ARCHIVE_DIR", str(tmp_path / "archive"))
    _write_active_precomputed_registry(tmp_path, model_key="custom_model")
    enqueue_identity_update("Gallery", "g-001")
    calls = []

    def fake_runner(**kwargs):
        calls.append(kwargs)
        registry = DLRegistry.reload()
        registry.pending_ids.clear()
        registry.save()
        return True, "ok"

    result = run_once(batch_size=1, runner=fake_runner)

    assert result.state == "completed"
    assert calls[0]["model_key"] == "custom_model"


def test_missing_incremental_baseline_requeues_and_reports_full_precompute_required(tmp_path, monkeypatch):
    monkeypatch.setenv("STARBOARD_ARCHIVE_DIR", str(tmp_path / "archive"))
    _write_active_precomputed_registry(tmp_path)
    enqueue_identity_update("Gallery", "g-001")

    result = run_once(
        batch_size=1,
        runner=lambda **kwargs: (False, "Incremental update requires baseline embeddings for all live IDs"),
    )

    assert result.state == "full_precompute_required"
    status = get_queue_status()
    assert status["queued"] == 1
    assert status["failed"] == 0
