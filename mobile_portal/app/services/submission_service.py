from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from fastapi import HTTPException, UploadFile, status

from ..config import get_settings
from ..adapters.id_policy import validate_target_mode
from ..adapters.ingest_adapter import UploadImage, ingest_images
from ..adapters.csv_adapter import write_metadata_row

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}


@dataclass
class SubmissionPayload:
    target_type: str
    target_mode: str
    target_id: str
    encounter_date: date
    encounter_suffix: str
    metadata: dict[str, str]
    client_notes: str = ''


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
        client_notes=raw.get('client_notes', ''),
    )


async def submit(payload_text: str, files: list[UploadFile]) -> dict:
    payload = parse_payload(payload_text)
    validate_target_mode(payload.target_type, payload.target_mode, payload.target_id)
    if not files:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='At least one image file is required')
    settings = get_settings()
    uploads = []
    for upload in files:
        suffix = Path(upload.filename or '').suffix.lower()
        if suffix not in IMAGE_EXTS:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f'Unsupported file type: {upload.filename}')
        content = await upload.read()
        if not content:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f'Empty file: {upload.filename}')
        if len(content) > settings.max_upload_mb * 1024 * 1024:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f'File exceeds max upload size: {upload.filename}')
        uploads.append(UploadImage(filename=upload.filename or 'upload.jpg', content=content))
    encounter_folder, written_paths = ingest_images(payload.target_type, payload.target_id, payload.encounter_date, payload.encounter_suffix, uploads)
    write_metadata_row(payload.target_type, payload.target_id, payload.metadata)
    return {
        'status': 'accepted',
        'entity_type': payload.target_type,
        'entity_id': payload.target_id,
        'encounter_folder': encounter_folder,
        'accepted_images': len(written_paths),
        'skipped_images': 0,
        'archive_paths_written': written_paths,
        'message': 'Submission incorporated into archive',
    }
