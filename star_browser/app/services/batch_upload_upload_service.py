from __future__ import annotations

from pathlib import Path
from uuid import uuid4
import os
import zipfile

from fastapi import HTTPException, UploadFile, status

from src.data.image_formats import is_importable_image

from ..config import get_settings
from ..models.batch_upload_api import BatchUploadUploadResponse


def _bundle_root(token: str) -> Path:
    return get_settings().staging_dir / token


def _scan_root(contents_dir: Path) -> Path:
    entries = list(contents_dir.iterdir()) if contents_dir.exists() else []
    dirs = [p for p in entries if p.is_dir()]
    top_level_images = [p for p in entries if p.is_file() and is_importable_image(p)]
    if len(dirs) == 1 and not top_level_images:
        child_entries = list(dirs[0].iterdir())
        child_top_level_images = [p for p in child_entries if p.is_file() and is_importable_image(p)]
        if not child_top_level_images:
            return dirs[0]
    return contents_dir


def resolve_uploaded_bundle_path(upload_token: str) -> Path | None:
    root = _bundle_root(upload_token) / 'contents'
    if root.exists() and root.is_dir():
        return _scan_root(root)
    return None


def _safe_extract_zip(zip_path: Path, dest_dir: Path) -> list[str]:
    root_entries: set[str] = set()
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            name = member.filename
            if not name or name.endswith('/'):
                continue
            if os.path.isabs(name):
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='absolute_paths_not_allowed')
            normalized = Path(name)
            if '..' in normalized.parts:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='path_traversal_not_allowed')
            if normalized.parts:
                root_entries.add(normalized.parts[0])
            target = dest_dir / normalized
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, open(target, 'wb') as dst:
                dst.write(src.read())
    return sorted(root_entries)


def stage_uploaded_bundle(file: UploadFile) -> BatchUploadUploadResponse:
    filename = file.filename or ''
    if not filename.lower().endswith('.zip'):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='zip_required')

    token = f'upload_{uuid4().hex[:12]}'
    bundle_root = _bundle_root(token)
    upload_dir = bundle_root / 'upload'
    contents_dir = bundle_root / 'contents'
    upload_dir.mkdir(parents=True, exist_ok=True)
    contents_dir.mkdir(parents=True, exist_ok=True)

    zip_path = upload_dir / 'bundle.zip'
    data = file.file.read()
    zip_path.write_bytes(data)
    root_entries = _safe_extract_zip(zip_path, contents_dir)

    file_count = sum(1 for p in contents_dir.rglob('*') if p.is_file())
    return BatchUploadUploadResponse(upload_token=token, file_count=file_count, root_entries=root_entries)
