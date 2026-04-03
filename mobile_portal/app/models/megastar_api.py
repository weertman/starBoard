from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .api import ImageDescriptor


class MegaStarLookupCandidate(BaseModel):
    rank: int
    entity_type: Literal['gallery'] = 'gallery'
    entity_id: str
    retrieval_score: float
    best_match_image: ImageDescriptor
    best_match_label: str | None = None
    encounter: str | None = None
    encounter_date: str | None = None


class MegaStarLookupResponse(BaseModel):
    query_image_name: str
    status: Literal['ok', 'unavailable']
    candidates: list[MegaStarLookupCandidate] = Field(default_factory=list)
    processing_ms: int = 0
    capability_state: Literal['enabled', 'disabled', 'unavailable'] | None = None
    availability_reason: str | None = None
