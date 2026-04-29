from __future__ import annotations

import json
import re
import tempfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from fastapi import HTTPException, UploadFile, status

from src.data.archive_paths import archive_root, metadata_csv_for, root_for
from src.data.csv_io import append_row
from src.data.id_registry import id_exists
from src.data.image_formats import IMPORT_IMAGE_EXTS, is_importable_image, normalized_suffix
from src.data.ingest import ensure_encounter_name, place_images
from src.data.validators import validate_id

IMAGE_EXTS = IMPORT_IMAGE_EXTS

ALLOWED_MODES = {
    ('query', 'create'),
    ('query', 'append'),
    ('gallery', 'append'),
}
SAFE_SUFFIX_RE = re.compile(r'^[A-Za-z0-9_-]*$')


@dataclass
class SubmissionPayload:
    target_type: str
    target_mode: str
    target_id: str
    encounter_date: date
    encounter_suffix: str
    metadata: dict[str, str]


def _canonical_target(target_type: str) -> str:
    return 'Gallery' if target_type == 'gallery' else 'Queries'


def _validate_target_mode(target_type: str, target_mode: str, target_id: str) -> None:
    if (target_type, target_mode) not in ALLOWED_MODES:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f'Unsupported target mode: {target_type}/{target_mode}')
    id_validation = validate_id(target_id)
    if not id_validation.ok:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=id_validation.message)
    if target_mode == 'append' and not id_exists(_canonical_target(target_type), target_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f'{target_type} ID not found: {target_id}')


def _validate_encounter_suffix(suffix: str) -> None:
    if not SAFE_SUFFIX_RE.fullmatch(suffix):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='encounter_suffix may contain letters, numbers, hyphen, and underscore only')


def parse_payload(payload_text: str) -> SubmissionPayload:
    try:
        raw = json.loads(payload_text)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='Invalid JSON payload') from exc
    try:
        encounter_date = date.fromisoformat(raw['encounter_date'])
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='encounter_date must be ISO YYYY-MM-DD') from exc
    return SubmissionPayload(
        target_type=raw['target_type'],
        target_mode=raw['target_mode'],
        target_id=raw['target_id'],
        encounter_date=encounter_date,
        encounter_suffix=raw.get('encounter_suffix', ''),
        metadata=dict(raw.get('metadata', {})),
    )


def _write_metadata_row(entity_type: str, entity_id: str, metadata: dict[str, str]) -> None:
    canonical = _canonical_target(entity_type)
    if entity_type == 'query':
        query_root = archive_root() / 'queries'
        query_root.mkdir(parents=True, exist_ok=True)
    csv_path, header = metadata_csv_for(canonical)
    row = dict(metadata)
    row['gallery_id' if entity_type == 'gallery' else 'query_id'] = entity_id
    append_row(csv_path, header, row)


def _target_root_for_submission(entity_type: str) -> Path:
    if entity_type == 'query':
        query_root = archive_root() / 'queries'
        query_root.mkdir(parents=True, exist_ok=True)
        return query_root
    target_root = root_for(_canonical_target(entity_type))
    target_root.mkdir(parents=True, exist_ok=True)
    return target_root


async def submit(payload_text: str, files: list[UploadFile]) -> dict:
    payload = parse_payload(payload_text)
    _validate_target_mode(payload.target_type, payload.target_mode, payload.target_id)
    _validate_encounter_suffix(payload.encounter_suffix)
    if not files:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='At least one image file is required')

    with tempfile.TemporaryDirectory(prefix='star_browser_upload_') as temp_dir:
        upload_paths: list[Path] = []
        temp_root = Path(temp_dir)
        for i, upload in enumerate(files):
            suffix = normalized_suffix(upload.filename or '')
            if not is_importable_image(upload.filename or ''):
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f'Unsupported file type: {upload.filename}')
            content = await upload.read()
            if not content:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f'Empty file: {upload.filename}')
            safe_name = Path(upload.filename or f'upload_{i}{suffix or ".jpg"}').name
            temp_path = temp_root / safe_name
            temp_path.write_bytes(content)
            upload_paths.append(temp_path)

        encounter_name = ensure_encounter_name(
            payload.encounter_date.year,
            payload.encounter_date.month,
            payload.encounter_date.day,
            payload.encounter_suffix,
        )
        target_root = _target_root_for_submission(payload.target_type)
        report = place_images(target_root, payload.target_id, encounter_name, upload_paths, move=False, observation_date=payload.encounter_date)
        _write_metadata_row(payload.target_type, payload.target_id, payload.metadata)
        return {
            'status': 'accepted',
            'entity_type': payload.target_type,
            'entity_id': payload.target_id,
            'encounter_folder': encounter_name,
            'accepted_images': len(report.ops),
            'skipped_images': max(0, len(upload_paths) - len(report.ops)),
            'archive_paths_written': [str(op.dest) for op in report.ops],
            'message': 'Submission incorporated into archive',
        }
