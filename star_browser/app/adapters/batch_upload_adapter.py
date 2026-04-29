from __future__ import annotations

from pathlib import Path

from src.data.id_registry import id_exists
from src.data.image_formats import is_importable_image
from src.data.ingest import (
    _encounter_suffix,
    _parse_encounter_date,
    detect_folder_depth,
    discover_grouped_ids_with_encounters,
    discover_ids_and_images,
    discover_ids_with_encounters,
)

from ..models.batch_upload_plan import PlannedBatchRow


def _transform_id(original_id: str, prefix: str = '', suffix: str = '') -> str:
    return f'{prefix}{original_id}{suffix}'


def _effective_mode(source_path: Path, requested_mode: str) -> str:
    if requested_mode == 'flat':
        return 'flat'
    detected = detect_folder_depth(source_path)
    if detected == 'ids':
        return 'encounters'
    if detected in {'single_id', 'grouped', 'flat'}:
        return detected
    if requested_mode in {'encounters', 'grouped'}:
        return requested_mode
    return 'empty'


def _single_id_encounters(source_path: Path) -> list[tuple[str, object, list[Path]]]:
    encounters = []
    for enc_dir in sorted(p for p in source_path.iterdir() if p.is_dir()):
        parsed = _parse_encounter_date(enc_dir.name)
        if parsed is None:
            continue
        files = sorted(p for p in enc_dir.rglob('*') if p.is_file() and is_importable_image(p))
        if files:
            encounters.append((enc_dir.name, parsed, files))
    return encounters


def discover_batch_source(
    source_path: Path,
    requested_mode: str,
    *,
    target_archive: str = 'gallery',
    id_prefix: str = '',
    id_suffix: str = '',
) -> list[PlannedBatchRow]:
    source_path = Path(source_path)
    rows: list[PlannedBatchRow] = []
    resolved = _effective_mode(source_path, requested_mode)

    target_name = 'Gallery' if target_archive == 'gallery' else 'Queries'

    if resolved == 'flat':
        items = discover_ids_and_images(source_path)
        for idx, (original_id, files) in enumerate(items):
            transformed = _transform_id(original_id, id_prefix, id_suffix)
            exists = id_exists(target_name, transformed)
            rows.append(PlannedBatchRow(
                row_id=f'row_{idx + 1:03d}',
                original_detected_id=original_id,
                transformed_target_id=transformed,
                action='append_existing' if exists else 'create_new',
                target_exists=exists,
                files=[Path(p) for p in files],
            ))
    elif resolved == 'single_id':
        original_id = source_path.name
        transformed = _transform_id(original_id, id_prefix, id_suffix)
        exists = id_exists(target_name, transformed)
        for row_num, (folder_name, dt, files) in enumerate(_single_id_encounters(source_path), start=1):
            rows.append(PlannedBatchRow(
                row_id=f'row_{row_num:03d}',
                original_detected_id=original_id,
                transformed_target_id=transformed,
                action='append_existing' if exists else 'create_new',
                target_exists=exists,
                encounter_folder_name=folder_name,
                encounter_date=dt.isoformat(),
                encounter_suffix=_encounter_suffix(folder_name) or None,
                files=[Path(p) for p in files],
            ))
    elif resolved == 'encounters':
        items = discover_ids_with_encounters(source_path)
        row_num = 0
        for original_id, encounters in items:
            transformed = _transform_id(original_id, id_prefix, id_suffix)
            exists = id_exists(target_name, transformed)
            for folder_name, dt, files in encounters:
                row_num += 1
                rows.append(PlannedBatchRow(
                    row_id=f'row_{row_num:03d}',
                    original_detected_id=original_id,
                    transformed_target_id=transformed,
                    action='append_existing' if exists else 'create_new',
                    target_exists=exists,
                    encounter_folder_name=folder_name,
                    encounter_date=dt.isoformat(),
                    encounter_suffix=_encounter_suffix(folder_name) or None,
                    files=[Path(p) for p in files],
                ))
    elif resolved == 'grouped':
        groups = discover_grouped_ids_with_encounters(source_path)
        row_num = 0
        for group_name, ids in groups:
            for original_id, encounters in ids:
                transformed = _transform_id(original_id, id_prefix, id_suffix)
                exists = id_exists(target_name, transformed)
                for folder_name, dt, files in encounters:
                    row_num += 1
                    rows.append(PlannedBatchRow(
                        row_id=f'row_{row_num:03d}',
                        original_detected_id=original_id,
                        transformed_target_id=transformed,
                        action='append_existing' if exists else 'create_new',
                        target_exists=exists,
                        group_name=group_name,
                        encounter_folder_name=folder_name,
                        encounter_date=dt.isoformat(),
                        encounter_suffix=_encounter_suffix(folder_name) or None,
                        files=[Path(p) for p in files],
                    ))
    return rows


def generate_plan_id() -> str:
    from uuid import uuid4
    return f'plan_{uuid4().hex[:12]}'
