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


class MetadataRow(BaseModel):
    row_index: int
    source: str
    values: dict[str, str]


class TimelineEvent(BaseModel):
    encounter: str
    date: str = ''
    label: str
    image_count: int = 0
    image_labels: list[str] = Field(default_factory=list)


class GalleryEntityResponse(BaseModel):
    archive_type: str = 'gallery'
    entity_id: str
    metadata_summary: dict[str, str]
    metadata_rows: list[MetadataRow] = Field(default_factory=list)
    timeline: list[TimelineEvent] = Field(default_factory=list)
    encounters: list[EncounterOption] = Field(default_factory=list)
    images: list[ImageDescriptor] = Field(default_factory=list)
