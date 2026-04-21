from __future__ import annotations

from pydantic import BaseModel, Field


class FirstOrderSearchRequest(BaseModel):
    query_id: str
    top_k: int = 10


class FirstOrderCandidate(BaseModel):
    entity_id: str
    score: float
    k_contrib: int
    field_breakdown: dict[str, float] = Field(default_factory=dict)


class FirstOrderSearchResponse(BaseModel):
    query_id: str
    candidates: list[FirstOrderCandidate] = Field(default_factory=list)
