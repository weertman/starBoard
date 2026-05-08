from __future__ import annotations

import csv
from pathlib import Path

from src.data import archive_paths as ap
from src.data.csv_io import last_row_per_id, normalize_id_value, read_rows_multi
from src.data.encounter_info import get_encounter_date, list_encounters_for_id
from src.data.id_registry import invalidate_id_cache, list_ids
from src.data.best_photo import reorder_files_with_best, save_best_for_id
from src.data.image_index import invalidate_image_cache, list_image_files
from src.data.observation_dates import last_observation_for_all
from src.data.rename_id import rename_id

from ..models.gallery_api import EncounterOption, IdReviewOption, ImageDescriptor, MetadataRow, TimelineEvent


def _target_name(archive_type: str) -> str:
    return 'Gallery' if archive_type == 'gallery' else 'Queries'


def _id_column(archive_type: str) -> str:
    return 'gallery_id' if archive_type == 'gallery' else 'query_id'


def _rewrite_matching_csv_rows(path: Path, match: callable, update: callable) -> int:
    if not path.exists():
        return 0
    with path.open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    updated = 0
    for row in rows:
        if match(row):
            before = dict(row)
            update(row, fieldnames)
            if row != before:
                updated += 1
    if updated == 0:
        return 0
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)
    return updated


def _rewrite_encounter_dates_id(target: str, old_id: str, new_id: str) -> int:
    path = ap.archive_root() / 'encounter_dates.csv'
    def matches(row: dict[str, str]) -> bool:
        row_target = (row.get('target') or row.get('archive_type') or '').strip()
        row_id = normalize_id_value(row.get('id') or row.get('entity_id') or row.get('gallery_id') or row.get('query_id') or '')
        return row_target == target and row_id == old_id
    def update(row: dict[str, str], _fieldnames: list[str]) -> None:
        if 'id' in row:
            row['id'] = new_id
        elif 'entity_id' in row:
            row['entity_id'] = new_id
        elif target == 'Gallery' and 'gallery_id' in row:
            row['gallery_id'] = new_id
        elif target == 'Queries' and 'query_id' in row:
            row['query_id'] = new_id
    return _rewrite_matching_csv_rows(path, matches, update)


def _enqueue_megastar_update(target: str, entity_id: str) -> None:
    try:
        from src.dl.megastar_queue import enqueue_identity_update
        enqueue_identity_update(target, entity_id, reason='id_review_edit', source='star_browser.id_review')
    except Exception:
        pass


def _metadata_summary_for_id(archive_type: str, entity_id: str) -> dict[str, str]:
    latest = _latest_metadata_by_id(archive_type)
    row = latest.get(entity_id, {})
    summary: dict[str, str] = {}
    id_col = _id_column(archive_type)
    for k, v in row.items():
        if k == id_col:
            continue
        text = (v or '').strip()
        if text:
            summary[k] = text
    return summary


def _latest_metadata_by_id(archive_type: str) -> dict[str, dict[str, str]]:
    paths = ap.metadata_csv_paths_for_read(_target_name(archive_type))
    rows = read_rows_multi(paths)
    return last_row_per_id(rows, _id_column(archive_type))


def _metadata_rows_for_id(archive_type: str, entity_id: str) -> list[MetadataRow]:
    id_col = _id_column(archive_type)
    rows: list[MetadataRow] = []
    for path in ap.metadata_csv_paths_for_read(_target_name(archive_type)):
        if not path.exists():
            continue
        with path.open(newline='', encoding='utf-8-sig') as f:
            for row_index, row in enumerate(csv.DictReader(f), start=1):
                if (row.get(id_col) or '').strip() != entity_id:
                    continue
                values = {
                    k: v.strip()
                    for k, v in row.items()
                    if k != id_col and v and v.strip()
                }
                rows.append(MetadataRow(row_index=row_index, source=path.name, values=values))
    return rows


def _encounter_options_for_id(archive_type: str, entity_id: str) -> list[EncounterOption]:
    items: list[EncounterOption] = []
    target = _target_name(archive_type)
    for encounter in list_encounters_for_id(target, entity_id):
        d = get_encounter_date(target, entity_id, encounter)
        date_text = d.isoformat() if d else ''
        label = f'{date_text} — {encounter}' if date_text else encounter
        items.append(EncounterOption(encounter=encounter, date=date_text, label=label))
    return items


def _image_files_for_id(archive_type: str, entity_id: str) -> list[Path]:
    target = _target_name(archive_type)
    if hasattr(list_image_files, 'cache_clear'):
        list_image_files.cache_clear()

    def sort_key(path: Path) -> tuple[int, str, str]:
        encounter = path.parent.name
        d = get_encounter_date(target, entity_id, encounter)
        return (1 if d else 0, d.isoformat() if d else '', path.name)

    files = sorted(list_image_files(target, entity_id), key=sort_key, reverse=True)
    return reorder_files_with_best(target, entity_id, list(files))


def _image_descriptors_for_id(archive_type: str, entity_id: str) -> list[ImageDescriptor]:
    descriptors: list[ImageDescriptor] = []
    for idx, path in enumerate(_image_files_for_id(archive_type, entity_id)):
        encounter = None
        try:
            encounter = path.parent.name
        except Exception:
            encounter = None
        image_id = f'{archive_type}:{entity_id}:{idx}'
        descriptors.append(
            ImageDescriptor(
                image_id=image_id,
                label=Path(path).name,
                encounter=encounter,
                preview_url=f'/api/id-review/media/{image_id}/preview',
                fullres_url=f'/api/id-review/media/{image_id}/full',
            )
        )
    return descriptors


def _timeline_for_id(encounters: list[EncounterOption], images: list[ImageDescriptor]) -> list[TimelineEvent]:
    labels_by_encounter: dict[str, list[str]] = {}
    for image in images:
        key = image.encounter or ''
        labels_by_encounter.setdefault(key, []).append(image.label)

    events: list[TimelineEvent] = []
    seen: set[str] = set()
    for encounter in encounters:
        seen.add(encounter.encounter)
        labels = labels_by_encounter.get(encounter.encounter, [])
        events.append(TimelineEvent(
            encounter=encounter.encounter,
            date=encounter.date,
            label=encounter.label,
            image_count=len(labels),
            image_labels=labels,
        ))
    for encounter, labels in labels_by_encounter.items():
        if encounter in seen:
            continue
        label = encounter or 'No encounter'
        events.append(TimelineEvent(
            encounter=encounter,
            date='',
            label=label,
            image_count=len(labels),
            image_labels=labels,
        ))
    return sorted(events, key=lambda event: (1 if event.date else 0, event.date or '', event.encounter), reverse=True)


def _option_label(entity_id: str, location: str, last_date: str) -> str:
    parts = [entity_id]
    if location:
        parts.append(location)
    if last_date:
        parts.append(last_date)
    return ' — '.join(parts)


def list_id_review_options(archive_type: str) -> list[IdReviewOption]:
    if archive_type not in {'gallery', 'query'}:
        return []
    target = _target_name(archive_type)
    latest = _latest_metadata_by_id(archive_type)
    last_dates = last_observation_for_all(target)
    invalidate_id_cache()
    options: list[IdReviewOption] = []
    for entity_id in list_ids(target, exclude_silent=(archive_type == 'query')):
        metadata = {
            k: (v or '').strip()
            for k, v in latest.get(entity_id, {}).items()
            if k != _id_column(archive_type) and v and v.strip()
        }
        location = metadata.get('location', '')
        observed = last_dates.get(entity_id)
        last_date = observed.isoformat() if observed else ''
        options.append(IdReviewOption(
            entity_id=entity_id,
            label=_option_label(entity_id, location, last_date),
            location=location,
            last_observation_date=last_date,
            metadata=metadata,
        ))
    return sorted(options, key=lambda option: (option.last_observation_date or '', option.entity_id), reverse=True)


def resolve_gallery_image_path(image_id: str) -> Path | None:
    return resolve_id_review_image_path(image_id)


def resolve_id_review_image_path(image_id: str) -> Path | None:
    parts = image_id.split(':')
    if len(parts) != 3 or parts[0] not in {'gallery', 'query'}:
        return None
    archive_type = parts[0]
    entity_id = parts[1]
    try:
        idx = int(parts[2])
    except ValueError:
        return None
    files = _image_files_for_id(archive_type, entity_id)
    if idx < 0 or idx >= len(files):
        return None
    return files[idx]


def set_id_review_best_image(image_id: str) -> tuple[bool, str, str, str, str]:
    parts = image_id.split(':')
    if len(parts) != 3 or parts[0] not in {'gallery', 'query'}:
        return False, 'id_review_image_not_found', '', '', ''
    archive_type = parts[0]
    entity_id = parts[1]
    path = resolve_id_review_image_path(image_id)
    if path is None:
        return False, 'id_review_image_not_found', archive_type, entity_id, ''
    target = _target_name(archive_type)
    save_best_for_id(target, entity_id, path)
    invalidate_image_cache()
    _enqueue_megastar_update(target, entity_id)
    return True, '', archive_type, entity_id, path.name


def load_gallery_entity(entity_id: str) -> tuple[dict[str, str], list[MetadataRow], list[TimelineEvent], list[EncounterOption], list[ImageDescriptor]]:
    return load_id_review_entity('gallery', entity_id)


def load_id_review_entity(archive_type: str, entity_id: str) -> tuple[dict[str, str], list[MetadataRow], list[TimelineEvent], list[EncounterOption], list[ImageDescriptor]]:
    if archive_type not in {'gallery', 'query'}:
        return {}, [], [], [], []
    encounters = _encounter_options_for_id(archive_type, entity_id)
    images = _image_descriptors_for_id(archive_type, entity_id)
    return (
        _metadata_summary_for_id(archive_type, entity_id),
        _metadata_rows_for_id(archive_type, entity_id),
        _timeline_for_id(encounters, images),
        encounters,
        images,
    )


def rename_id_review_entity(archive_type: str, old_id: str, new_id: str) -> tuple[bool, list[str], str]:
    if archive_type not in {'gallery', 'query'}:
        return False, ['archive_type_not_found'], old_id
    target = _target_name(archive_type)
    old_id = normalize_id_value(old_id)
    new_id = normalize_id_value(new_id)
    report = rename_id(target, old_id, new_id)
    if not report.success:
        return False, report.errors, old_id
    _rewrite_encounter_dates_id(target, old_id, new_id)
    invalidate_id_cache()
    invalidate_image_cache()
    _enqueue_megastar_update(target, new_id)
    return True, [], new_id


def update_id_review_metadata(archive_type: str, entity_id: str, metadata: dict[str, str]) -> tuple[bool, str]:
    if archive_type not in {'gallery', 'query'}:
        return False, 'archive_type_not_found'
    target = _target_name(archive_type)
    id_col = _id_column(archive_type)
    entity_id = normalize_id_value(entity_id)
    csv_path, _header = ap.metadata_csv_for(target)
    clean = {str(k).strip(): str(v).strip() for k, v in metadata.items() if str(k).strip() and str(k).strip() != id_col}

    def matches(row: dict[str, str]) -> bool:
        return normalize_id_value(row.get(id_col, '')) == entity_id

    def update(row: dict[str, str], fieldnames: list[str]) -> None:
        for key, value in clean.items():
            if key not in fieldnames:
                fieldnames.append(key)
            row[key] = value

    updated = _rewrite_matching_csv_rows(csv_path, matches, update)
    if updated == 0:
        return False, 'metadata_row_not_found'
    invalidate_id_cache()
    _enqueue_megastar_update(target, entity_id)
    return True, ''
