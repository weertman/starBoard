import pytest
from fastapi import HTTPException

from mobile_portal.app.adapters.id_policy import validate_target_mode


def test_query_create_allowed():
    validate_target_mode('query', 'create', 'q1')


def test_gallery_create_rejected():
    with pytest.raises(HTTPException) as exc:
        validate_target_mode('gallery', 'create', 'g1')
    assert exc.value.status_code == 400
