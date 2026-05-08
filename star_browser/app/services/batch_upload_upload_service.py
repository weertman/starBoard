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


def _extract_staged_zip(token: str) -> BatchUploadUploadResponse:
    bundle_root = _bundle_root(token)
    zip_path = bundle_root / 'upload' / 'bundle.zip'
    contents_dir = bundle_root / 'contents'
    if not zip_path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='upload_token_not_found')
    contents_dir.mkdir(parents=True, exist_ok=True)
    root_entries = _safe_extract_zip(zip_path, contents_dir)
    file_count = sum(1 for p in contents_dir.rglob('*') if p.is_file())
    return BatchUploadUploadResponse(upload_token=token, file_count=file_count, root_entries=root_entries)


def stage_uploaded_bundle(file: UploadFile) -> BatchUploadUploadResponse:
    filename = file.filename or ''
    if not filename.lower().endswith('.zip'):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='zip_required')

    token = f'upload_{uuid4().hex[:12]}'
    bundle_root = _bundle_root(token)
    upload_dir = bundle_root / 'upload'
    upload_dir.mkdir(parents=True, exist_ok=True)

    zip_path = upload_dir / 'bundle.zip'
    zip_path.write_bytes(file.file.read())
    return _extract_staged_zip(token)


def start_chunked_uploaded_bundle() -> dict[str, str]:
    token = f'upload_{uuid4().hex[:12]}'
    upload_dir = _bundle_root(token) / 'upload'
    upload_dir.mkdir(parents=True, exist_ok=True)
    return {'upload_token': token}


def _validate_upload_token(token: str) -> None:
    if not token.startswith('upload_') or '/' in token or '..' in token:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='invalid_upload_token')


def append_uploaded_zip_chunk(token: str, offset: int, total_size: int, filename: str, chunk: UploadFile) -> None:
    _validate_upload_token(token)
    if offset < 0 or total_size <= 0 or offset > total_size:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='invalid_chunk_offset')
    if not filename.lower().endswith('.zip'):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='zip_required')
    upload_dir = _bundle_root(token) / 'upload'
    if not upload_dir.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='upload_token_not_found')

    zip_path = upload_dir / 'bundle.zip'
    data = chunk.file.read()
    if offset + len(data) > total_size:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='chunk_exceeds_total_size')
    mode = 'r+b' if zip_path.exists() else 'wb'
    with open(zip_path, mode) as dst:
        size = dst.seek(0, os.SEEK_END)
        if size != offset:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail='chunk_offset_mismatch')
        dst.write(data)


def finalize_chunked_uploaded_bundle(token: str) -> BatchUploadUploadResponse:
    _validate_upload_token(token)
    zip_path = _bundle_root(token) / 'upload' / 'bundle.zip'
    if not zip_path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='upload_token_not_found')
    if not zipfile.is_zipfile(zip_path):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='invalid_zip')
    return _extract_staged_zip(token)


def _safe_uploaded_relative_path(name: str) -> Path:
    if not name or os.path.isabs(name):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='invalid_relative_path')
    normalized = Path(name)
    if '..' in normalized.parts:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='path_traversal_not_allowed')
    return normalized


def stage_uploaded_folder(files: list[UploadFile]) -> BatchUploadUploadResponse:
    if not files:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='files_required')

    token = f'upload_{uuid4().hex[:12]}'
    contents_dir = _bundle_root(token) / 'contents'
    contents_dir.mkdir(parents=True, exist_ok=True)

    root_entries: set[str] = set()
    for file in files:
        relative_path = _safe_uploaded_relative_path(file.filename or '')
        if relative_path.parts:
            root_entries.add(relative_path.parts[0])
        target = contents_dir / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(file.file.read())

    file_count = sum(1 for p in contents_dir.rglob('*') if p.is_file())
    return BatchUploadUploadResponse(upload_token=token, file_count=file_count, root_entries=sorted(root_entries))
