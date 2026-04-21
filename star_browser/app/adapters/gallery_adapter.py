from __future__ import annotations

from pathlib import Path

from src.data import archive_paths as ap
from src.data.csv_io import last_row_per_id, read_rows_multi
from src.data.encounter_info import get_encounter_date, list_encounters_for_id
from src.data.image_index import list_image_files

from ..models.gallery_api import EncounterOption, ImageDescriptor


def _metadata_summary_for_gallery(entity_id: str) -> dict[str, str]:
    paths = ap.metadata_csv_paths_for_read('Gallery')
    rows = read_rows_multi(paths)
    latest = last_row_per_id(rows, 'gallery_id')
    row = latest.get(entity_id, {})
    summary: dict[str, str] = {}
    for k, v in row.items():
        if k == 'gallery_id':
            continue
        text = (v or '').strip()
        if text:
            summary[k] = text
    return summary


def _encounter_options_for_gallery(entity_id: str) -> list[EncounterOption]:
    items: list[EncounterOption] = []
    for encounter in list_encounters_for_id('Gallery', entity_id):
        d = get_encounter_date('Gallery', entity_id, encounter)
        date_text = d.isoformat() if d else ''
        label = f'{date_text} — {encounter}' if date_text else encounter
        items.append(EncounterOption(encounter=encounter, date=date_text, label=label))
    return items


def _image_descriptors_for_gallery(entity_id: str) -> list[ImageDescriptor]:
    descriptors: list[ImageDescriptor] = []
    for idx, path in enumerate(list_image_files('Gallery', entity_id)):
        encounter = None
        try:
            encounter = path.parent.name
        except Exception:
            encounter = None
        image_id = f'gallery:{entity_id}:{idx}'
        descriptors.append(
            ImageDescriptor(
                image_id=image_id,
                label=Path(path).name,
                encounter=encounter,
                preview_url=f'/api/gallery/media/{image_id}/preview',
                fullres_url=f'/api/gallery/media/{image_id}/full',
            )
        )
    return descriptors


def resolve_gallery_image_path(image_id: str) -> Path | None:
    parts = image_id.split(':')
    if len(parts) != 3 or parts[0] != 'gallery':
        return None
    entity_id = parts[1]
    try:
        idx = int(parts[2])
    except ValueError:
        return None
    files = list(list_image_files('Gallery', entity_id))
    if idx < 0 or idx >= len(files):
        return None
    return files[idx]


def load_gallery_entity(entity_id: str) -> tuple[dict[str, str], list[EncounterOption], list[ImageDescriptor]]:
    return (
        _metadata_summary_for_gallery(entity_id),
        _encounter_options_for_gallery(entity_id),
        _image_descriptors_for_gallery(entity_id),
    )
