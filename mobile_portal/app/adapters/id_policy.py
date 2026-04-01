from __future__ import annotations

from fastapi import HTTPException, status

from .archive_paths import entity_exists


ALLOWED_MODES = {
    ('query', 'create'),
    ('query', 'append'),
    ('gallery', 'append'),
}


def validate_target_mode(target_type: str, target_mode: str, target_id: str | None) -> None:
    if (target_type, target_mode) not in ALLOWED_MODES:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f'Unsupported target mode: {target_type}/{target_mode}')
    if target_mode == 'append':
        if not target_id:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='target_id is required for append mode')
        if not entity_exists(target_type, target_id):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f'{target_type} ID not found: {target_id}')
    if target_mode == 'create' and not target_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='target_id is required for create mode in v1')
