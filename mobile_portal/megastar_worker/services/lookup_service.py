from __future__ import annotations

from functools import lru_cache

from mobile_portal.app.services.megastar_lookup_service import MegaStarLookupService
from mobile_portal.megastar_worker.config import as_portal_settings, get_settings


@lru_cache(maxsize=1)
def get_megastar_worker_lookup_service() -> MegaStarLookupService:
    settings = get_settings()
    return MegaStarLookupService(settings=as_portal_settings(settings))
