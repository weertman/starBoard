from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field


class MegaStarCapabilityInfo(BaseModel):
    enabled: bool
    state: Literal['enabled', 'disabled', 'unavailable']
    backend: Literal['local', 'worker']
    reason: str | None = None
    model_key: str | None = None


class SessionResponse(BaseModel):
    authenticated_email: str
    capabilities: dict[str, bool]
    megastar_lookup: MegaStarCapabilityInfo | None = None


class SchemaFieldOption(BaseModel):
    label: str
    value: Any


class SchemaField(BaseModel):
    name: str
    display_name: str
    field_type: str
    group: str
    group_display_name: str
    required: bool
    tooltip: str = ''
    min_value: float | None = None
    max_value: float | None = None
    options: list[SchemaFieldOption] = Field(default_factory=list)
    vocabulary: list[str] = Field(default_factory=list)
    mobile_widget: str


class MetadataSchemaResponse(BaseModel):
    fields: list[SchemaField]


class ImageDescriptor(BaseModel):
    image_id: str
    label: str
    encounter: str | None = None
    fullres_url: str
    preview_url: str
    width: int | None = None
    height: int | None = None


class EncounterOption(BaseModel):
    encounter: str
    date: str = ''
    label: str


class EncounterOptionsResponse(BaseModel):
    entity_type: Literal['gallery', 'query']
    entity_id: str
    encounters: list[EncounterOption] = Field(default_factory=list)


class ImageWindow(BaseModel):
    offset: int
    count: int
    total: int
    items: list[ImageDescriptor]
    next_offset: int | None = None


class ArchiveEntityResponse(BaseModel):
    entity_type: Literal['gallery', 'query']
    entity_id: str
    metadata_summary: dict[str, Any]
    encounters: list[EncounterOption] = Field(default_factory=list)
    selected_encounter: str = ''
    image_window: ImageWindow


class ImageWindowResponse(ImageWindow):
    pass


class EntitySuggestionResponse(BaseModel):
    entity_type: Literal['gallery', 'query']
    query: str
    items: list[str]


class LookupOptionsResponse(BaseModel):
    entity_type: Literal['gallery', 'query']
    location: str
    locations: list[str]
    ids: list[str]


class SubmissionResponse(BaseModel):
    status: str
    entity_type: Literal['gallery', 'query']
    entity_id: str
    encounter_folder: str
    accepted_images: int
    skipped_images: int
    archive_paths_written: list[str]
    message: str
