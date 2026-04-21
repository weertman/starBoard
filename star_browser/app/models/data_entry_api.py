from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


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


class SubmissionResponse(BaseModel):
    status: str
    entity_type: Literal['gallery', 'query']
    entity_id: str
    encounter_folder: str
    accepted_images: int
    skipped_images: int
    archive_paths_written: list[str]
    message: str
