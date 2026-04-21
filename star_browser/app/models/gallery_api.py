from __future__ import annotations

from pydantic import BaseModel, Field


class ImageDescriptor(BaseModel):
    image_id: str
    label: str
    encounter: str | None = None
    preview_url: str
    fullres_url: str


class EncounterOption(BaseModel):
    encounter: str
    date: str = ''
    label: str


class GalleryEntityResponse(BaseModel):
    entity_id: str
    metadata_summary: dict[str, str]
    encounters: list[EncounterOption] = Field(default_factory=list)
    images: list[ImageDescriptor] = Field(default_factory=list)
