from __future__ import annotations

from pathlib import Path
import hashlib
import mimetypes

from fastapi import HTTPException, status
from fastapi.responses import FileResponse
from PIL import Image

from ..config import get_settings
from ..adapters.image_manifest_adapter import resolve_image_path


PREVIEW_MAX = 1600


def _preview_cache_path(image_path: Path) -> Path:
    settings = get_settings()
    settings.preview_cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(str(image_path).encode('utf-8')).hexdigest()[:24]
    return settings.preview_cache_dir / f'{key}.jpg'


def get_full_image_response(image_id: str) -> FileResponse:
    try:
        path = resolve_image_path(image_id)
    except (ValueError, IndexError):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Image not found')
    media_type = mimetypes.guess_type(path.name)[0] or 'application/octet-stream'
    return FileResponse(path, media_type=media_type, filename=path.name)


def get_preview_image_response(image_id: str) -> FileResponse:
    try:
        path = resolve_image_path(image_id)
    except (ValueError, IndexError):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Image not found')
    cached = _preview_cache_path(path)
    if not cached.exists() or cached.stat().st_mtime < path.stat().st_mtime:
        with Image.open(path) as img:
            img = img.convert('RGB')
            img.thumbnail((PREVIEW_MAX, PREVIEW_MAX))
            img.save(cached, format='JPEG', quality=85)
    return FileResponse(cached, media_type='image/jpeg', filename=f'{path.stem}_preview.jpg')
