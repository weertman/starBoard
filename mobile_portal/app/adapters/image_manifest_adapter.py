from __future__ import annotations

from pathlib import Path
from collections import OrderedDict

from PIL import Image

from src.data.image_index import list_image_files
from src.data.encounter_info import get_encounter_date

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}


def list_entity_images(entity_type: str, entity_id: str) -> list[Path]:
    canonical = 'Gallery' if entity_type == 'gallery' else 'Queries'
    return list(list_image_files(canonical, entity_id))


def list_entity_encounters(entity_type: str, entity_id: str) -> list[dict]:
    canonical = 'Gallery' if entity_type == 'gallery' else 'Queries'
    images = list_entity_images(entity_type, entity_id)
    encounters: OrderedDict[str, dict] = OrderedDict()
    for image in images:
        try:
            encounter_name = image.parent.name
        except Exception:
            continue
        if encounter_name not in encounters:
            date_value = get_encounter_date(canonical, entity_id, encounter_name)
            encounters[encounter_name] = {
                'encounter': encounter_name,
                'date': date_value.isoformat() if date_value else '',
                'label': f"{encounter_name} ({date_value.isoformat() if date_value else 'unknown date'})",
            }
    return list(encounters.values())


def build_image_id(entity_type: str, entity_id: str, index: int) -> str:
    return f'{entity_type}:{entity_id}:{index}'


def parse_image_id(image_id: str) -> tuple[str, str, int]:
    try:
        entity_type, entity_id, index_str = image_id.split(':', 2)
        return entity_type, entity_id, int(index_str)
    except Exception as exc:
        raise ValueError(f'Invalid image_id: {image_id}') from exc


def image_descriptor(entity_type: str, entity_id: str, index: int, image_path: Path) -> dict:
    width = height = None
    try:
        with Image.open(image_path) as img:
            width, height = img.size
    except Exception:
        pass
    iid = build_image_id(entity_type, entity_id, index)
    encounter_name = image_path.parent.name if image_path.parent else ''
    return {
        'image_id': iid,
        'label': image_path.name,
        'encounter': encounter_name,
        'fullres_url': f'/api/archive/media/{iid}/full',
        'preview_url': f'/api/archive/media/{iid}/preview',
        'width': width,
        'height': height,
    }


def window_for_entity(entity_type: str, entity_id: str, offset: int, limit: int, encounter: str | None = None) -> dict:
    images = list_entity_images(entity_type, entity_id)
    indexed = list(enumerate(images))
    if encounter:
        indexed = [(idx, p) for idx, p in indexed if p.parent.name == encounter]
    sliced = indexed[offset:offset+limit]
    items = [image_descriptor(entity_type, entity_id, idx, p) for idx, p in sliced]
    next_offset = offset + len(items)
    return {
        'offset': offset,
        'count': len(items),
        'total': len(indexed),
        'items': items,
        'next_offset': next_offset if next_offset < len(indexed) else None,
    }


def resolve_image_path(image_id: str) -> Path:
    entity_type, entity_id, index = parse_image_id(image_id)
    images = list_entity_images(entity_type, entity_id)
    if index < 0 or index >= len(images):
        raise IndexError(image_id)
    return images[index]
