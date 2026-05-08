from __future__ import annotations

from src.dl.megastar_artifact_lock import megastar_artifact_lock, lock_path


def test_artifact_lock_blocks_second_acquire(tmp_path, monkeypatch):
    monkeypatch.setenv("STARBOARD_ARCHIVE_DIR", str(tmp_path / "archive"))

    with megastar_artifact_lock(timeout_seconds=0.1):
        assert lock_path().exists()
        try:
            with megastar_artifact_lock(timeout_seconds=0.01):
                raise AssertionError("second lock acquire should not succeed")
        except TimeoutError:
            pass
