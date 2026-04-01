from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
import tempfile

from src.data.archive_paths import root_for
from src.data.ingest import ensure_encounter_name, place_images


@dataclass
class UploadImage:
    filename: str
    content: bytes


def ingest_images(entity_type: str, entity_id: str, encounter_date: date, encounter_suffix: str, files: list[UploadImage]) -> tuple[str, list[str]]:
    canonical = 'Gallery' if entity_type == 'gallery' else 'Queries'
    encounter_name = ensure_encounter_name(encounter_date.year, encounter_date.month, encounter_date.day, encounter_suffix or '')
    target_root = root_for(canonical)
    target_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='mobile_portal_upload_') as tmpdir:
        tmp_paths = []
        for idx, upload in enumerate(files):
            suffix = Path(upload.filename).suffix or '.jpg'
            safe_name = Path(upload.filename).name or f'upload_{idx}{suffix}'
            path = Path(tmpdir) / safe_name
            path.write_bytes(upload.content)
            tmp_paths.append(path)
        report = place_images(target_root, entity_id, encounter_name, tmp_paths, move=False, observation_date=encounter_date)
    return encounter_name, [str(op.dest) for op in report.ops]
