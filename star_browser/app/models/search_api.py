from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class FirstOrderSearchRequest(BaseModel):
    query_id: str
    top_k: int = 10
    preset: Literal['all', 'colors', 'text', 'arms_patterns'] = 'all'


class FirstOrderCandidate(BaseModel):
    entity_id: str
    score: float
    k_contrib: int
    field_breakdown: dict[str, float] = Field(default_factory=dict)


class FirstOrderSearchResponse(BaseModel):
    query_id: str
    preset: Literal['all', 'colors', 'text', 'arms_patterns']
    candidates: list[FirstOrderCandidate] = Field(default_factory=list)
