from __future__ import annotations

from fastapi import APIRouter, Depends

from ..auth import require_authenticated_email
from ..models.data_entry_api import LocationSitesResponse
from ..services.location_service import get_location_sites

router = APIRouter()


@router.get('/locations/sites', response_model=LocationSitesResponse)
def location_sites(_user_email: str = Depends(require_authenticated_email)):
    return get_location_sites()
