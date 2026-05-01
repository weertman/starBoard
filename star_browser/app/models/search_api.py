from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


QueryWorkflowState = Literal['not_attempted', 'pinned', 'attempted', 'matched']


class FirstOrderQueryOption(BaseModel):
    query_id: str
    state: QueryWorkflowState = 'not_attempted'
    last_observation_date: str | None = None
    last_location: str | None = None
    easy_match_score: float = 0.0
    quality: dict[str, float | None] = Field(default_factory=dict)


class FirstOrderQueryOptionsResponse(BaseModel):
    queries: list[FirstOrderQueryOption] = Field(default_factory=list)


class FirstOrderSearchRequest(BaseModel):
    query_id: str
    top_k: int = 10
    preset: Literal['all', 'colors', 'text', 'arms_patterns', 'megastar'] = 'all'
    query_image_id: str | None = None


class FirstOrderCandidate(BaseModel):
    entity_id: str
    score: float
    k_contrib: int
    field_breakdown: dict[str, float] = Field(default_factory=dict)
    preferred_image_id: str | None = None


class FirstOrderSearchResponse(BaseModel):
    query_id: str
    preset: Literal['all', 'colors', 'text', 'arms_patterns', 'megastar']
    query_image_id: str | None = None
    candidates: list[FirstOrderCandidate] = Field(default_factory=list)


class FirstOrderMediaImage(BaseModel):
    image_id: str
    label: str
    encounter: str | None = None
    preview_url: str
    fullres_url: str
    is_best: bool = False


class FirstOrderMediaResponse(BaseModel):
    target_type: Literal['query', 'gallery']
    entity_id: str
    images: list[FirstOrderMediaImage] = Field(default_factory=list)
