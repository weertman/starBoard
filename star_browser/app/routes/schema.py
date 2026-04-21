from __future__ import annotations

from fastapi import APIRouter, Depends

from ..auth import require_authenticated_email
from ..models.data_entry_api import MetadataSchemaResponse
from ..services.schema_service import get_metadata_schema

router = APIRouter()


@router.get('/schema/metadata', response_model=MetadataSchemaResponse)
def metadata_schema(_user_email: str = Depends(require_authenticated_email)):
    return get_metadata_schema()
