from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Literal

from PIL import Image
from fastapi import HTTPException, status
from fastapi.responses import Response

from src.data.best_photo import find_best_index, reorder_files_with_best
from src.data.image_index import list_image_files

from ..models.search_api import FirstOrderMediaImage, FirstOrderMediaResponse

TargetType = Literal['query', 'gallery']


def _canonical_target(target_type: TargetType) -> str:
    return 'Queries' if target_type == 'query' else 'Gallery'


def _media_prefix(target_type: TargetType) -> str:
    return 'query' if target_type == 'query' else 'gallery'


def _image_id(target_type: TargetType, entity_id: str, idx: int) -> str:
    return f'{_media_prefix(target_type)}:{entity_id}:{idx}'


def _encounter_for_path(path: Path) -> str | None:
    try:
        return path.parent.name
    except Exception:
        return None


def list_first_order_media(target_type: TargetType, entity_id: str) -> FirstOrderMediaResponse:
    target = _canonical_target(target_type)
    original_files = list(list_image_files(target, entity_id))
    if not original_files:
        return FirstOrderMediaResponse(target_type=target_type, entity_id=entity_id, images=[])

    best_path = original_files[find_best_index(target, entity_id, original_files)]
    files = reorder_files_with_best(target, entity_id, original_files)
    images: list[FirstOrderMediaImage] = []
    for idx, path in enumerate(files):
        image_id = _image_id(target_type, entity_id, idx)
        try:
            is_best = path.resolve() == best_path.resolve()
        except Exception:
            is_best = idx == 0
        images.append(
            FirstOrderMediaImage(
                image_id=image_id,
                label=path.name,
                encounter=_encounter_for_path(path),
                preview_url=f'/api/first-order/media/{image_id}/preview',
                fullres_url=f'/api/first-order/media/{image_id}/full',
                is_best=is_best,
            )
        )
    return FirstOrderMediaResponse(target_type=target_type, entity_id=entity_id, images=images)


def resolve_first_order_media_path(image_id: str) -> Path | None:
    parts = image_id.split(':')
    if len(parts) != 3 or parts[0] not in {'query', 'gallery'}:
        return None
    target_type: TargetType = 'query' if parts[0] == 'query' else 'gallery'
    entity_id = parts[1]
    try:
        idx = int(parts[2])
    except ValueError:
        return None
    files = list_first_order_media(target_type, entity_id).images
    if idx < 0 or idx >= len(files):
        return None
    # Resolve against the same best-first order used to create descriptors.
    target = _canonical_target(target_type)
    ordered_paths = reorder_files_with_best(target, entity_id, list(list_image_files(target, entity_id)))
    if idx >= len(ordered_paths):
        return None
    return ordered_paths[idx]


def resized_preview_response(path: Path, *, long_edge: int = 48) -> Response:
    try:
        with Image.open(path) as image:
            image = image.convert('RGB')
            image.thumbnail((long_edge, long_edge))
            buf = BytesIO()
            image.save(buf, format='JPEG', quality=60, optimize=True)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='first_order_image_not_readable') from exc
    return Response(content=buf.getvalue(), media_type='image/jpeg')
