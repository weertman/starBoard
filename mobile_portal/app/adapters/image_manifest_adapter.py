from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PIL import Image

from src.data.image_index import list_image_files

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}


def list_entity_images(entity_type: str, entity_id: str) -> list[Path]:
    canonical = 'Gallery' if entity_type == 'gallery' else 'Queries'
    return list(list_image_files(canonical, entity_id))


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
    return {
        'image_id': iid,
        'label': image_path.name,
        'fullres_url': f'/api/archive/media/{iid}/full',
        'preview_url': f'/api/archive/media/{iid}/preview',
        'width': width,
        'height': height,
    }


def window_for_entity(entity_type: str, entity_id: str, offset: int, limit: int) -> dict:
    images = list_entity_images(entity_type, entity_id)
    items = [image_descriptor(entity_type, entity_id, idx, p) for idx, p in enumerate(images[offset:offset+limit], start=offset)]
    next_offset = offset + len(items)
    return {
        'offset': offset,
        'count': len(items),
        'total': len(images),
        'items': items,
        'next_offset': next_offset if next_offset < len(images) else None,
    }


def resolve_image_path(image_id: str) -> Path:
    entity_type, entity_id, index = parse_image_id(image_id)
    images = list_entity_images(entity_type, entity_id)
    if index < 0 or index >= len(images):
        raise IndexError(image_id)
    return images[index]
