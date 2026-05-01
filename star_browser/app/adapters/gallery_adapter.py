from __future__ import annotations

from pathlib import Path

from src.data import archive_paths as ap
from src.data.csv_io import last_row_per_id, read_rows_multi
from src.data.encounter_info import get_encounter_date, list_encounters_for_id
from src.data.image_index import list_image_files

from ..models.gallery_api import EncounterOption, ImageDescriptor


def _target_name(archive_type: str) -> str:
    return 'Gallery' if archive_type == 'gallery' else 'Queries'


def _id_column(archive_type: str) -> str:
    return 'gallery_id' if archive_type == 'gallery' else 'query_id'


def _metadata_summary_for_id(archive_type: str, entity_id: str) -> dict[str, str]:
    paths = ap.metadata_csv_paths_for_read(_target_name(archive_type))
    rows = read_rows_multi(paths)
    id_col = _id_column(archive_type)
    latest = last_row_per_id(rows, id_col)
    row = latest.get(entity_id, {})
    summary: dict[str, str] = {}
    for k, v in row.items():
        if k == id_col:
            continue
        text = (v or '').strip()
        if text:
            summary[k] = text
    return summary


def _encounter_options_for_id(archive_type: str, entity_id: str) -> list[EncounterOption]:
    items: list[EncounterOption] = []
    target = _target_name(archive_type)
    for encounter in list_encounters_for_id(target, entity_id):
        d = get_encounter_date(target, entity_id, encounter)
        date_text = d.isoformat() if d else ''
        label = f'{date_text} — {encounter}' if date_text else encounter
        items.append(EncounterOption(encounter=encounter, date=date_text, label=label))
    return items


def _image_descriptors_for_id(archive_type: str, entity_id: str) -> list[ImageDescriptor]:
    descriptors: list[ImageDescriptor] = []
    target = _target_name(archive_type)
    for idx, path in enumerate(list_image_files(target, entity_id)):
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
    files = list(list_image_files(_target_name(archive_type), entity_id))
    if idx < 0 or idx >= len(files):
        return None
    return files[idx]


def load_gallery_entity(entity_id: str) -> tuple[dict[str, str], list[EncounterOption], list[ImageDescriptor]]:
    return load_id_review_entity('gallery', entity_id)


def load_id_review_entity(archive_type: str, entity_id: str) -> tuple[dict[str, str], list[EncounterOption], list[ImageDescriptor]]:
    if archive_type not in {'gallery', 'query'}:
        return {}, [], []
    return (
        _metadata_summary_for_id(archive_type, entity_id),
        _encounter_options_for_id(archive_type, entity_id),
        _image_descriptors_for_id(archive_type, entity_id),
    )
