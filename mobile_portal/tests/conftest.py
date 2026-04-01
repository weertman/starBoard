from __future__ import annotations

import importlib
from pathlib import Path

from PIL import Image
from fastapi.testclient import TestClient


def build_test_app(tmp_path, monkeypatch):
    archive = tmp_path / 'archive'
    (archive / 'gallery').mkdir(parents=True, exist_ok=True)
    (archive / 'queries').mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv('STARBOARD_ARCHIVE_DIR', str(archive))
    monkeypatch.setenv('STARBOARD_MOBILE_PREVIEW_CACHE_DIR', str(tmp_path / 'preview_cache'))
    import mobile_portal.app.main as main_mod
    importlib.reload(main_mod)
    return main_mod.create_app(), archive


def make_image(path: Path, color=(200, 100, 50)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new('RGB', (120, 90), color)
    img.save(path)
