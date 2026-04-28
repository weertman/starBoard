from __future__ import annotations

from pydantic import BaseModel


class MegaStarCapabilityInfo(BaseModel):
    enabled: bool
    state: str
    backend: str
    reason: str | None = None
    model_key: str | None = None


class SessionResponse(BaseModel):
    authenticated_email: str
    capabilities: dict[str, bool]
    megastar_lookup: MegaStarCapabilityInfo | None = None
