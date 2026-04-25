import pytest
from fastapi import HTTPException

from mobile_portal.app.adapters.id_policy import validate_target_mode


def test_query_create_allowed():
    validate_target_mode('query', 'create', 'q1')


def test_gallery_create_allowed():
    validate_target_mode('gallery', 'create', 'g1')


def test_create_requires_target_id():
    with pytest.raises(HTTPException) as exc:
        validate_target_mode('gallery', 'create', '')
    assert exc.value.status_code == 400


def test_append_requires_existing_target_id():
    with pytest.raises(HTTPException) as exc:
        validate_target_mode('gallery', 'append', 'missing-gallery-id')
    assert exc.value.status_code == 404
