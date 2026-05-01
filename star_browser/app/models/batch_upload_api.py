from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class UploadedBundleImportSource(BaseModel):
    type: Literal['uploaded_bundle']
    upload_token: str


BatchUploadImportSource = UploadedBundleImportSource


class BatchUploadLocationDraft(BaseModel):
    location: str = ''
    latitude: str = ''
    longitude: str = ''


class BatchUploadDiscoverRequest(BaseModel):
    target_archive: Literal['gallery', 'query']
    discovery_mode: Literal['auto', 'flat', 'encounters', 'grouped']
    id_prefix: str = ''
    id_suffix: str = ''
    flat_encounter_date: str = ''
    flat_encounter_suffix: str = ''
    batch_location: BatchUploadLocationDraft = Field(default_factory=BatchUploadLocationDraft)
    import_source: BatchUploadImportSource


class BatchUploadWarning(BaseModel):
    code: str
    message: str
    row_id: str | None = None


class BatchUploadDiscoverRow(BaseModel):
    row_id: str
    original_detected_id: str
    transformed_target_id: str
    action: Literal['create_new', 'append_existing', 'skip']
    target_exists: bool
    group_name: str | None = None
    encounter_folder_name: str | None = None
    encounter_date: str | None = None
    encounter_suffix: str | None = None
    image_count: int
    sample_labels: list[str] = Field(default_factory=list)
    source_ref: str
    warnings: list[BatchUploadWarning] = Field(default_factory=list)


class BatchUploadDiscoverSummary(BaseModel):
    detected_rows: int
    detected_ids: int
    total_images: int
    new_ids: int
    existing_ids: int
    warnings: int
    errors: int


class BatchUploadDiscoverResponse(BaseModel):
    plan_id: str
    target_archive: Literal['gallery', 'query']
    requested_discovery_mode: Literal['auto', 'flat', 'encounters', 'grouped']
    resolved_discovery_mode: Literal['flat', 'encounters', 'grouped', 'single_id', 'empty']
    summary: BatchUploadDiscoverSummary
    rows: list[BatchUploadDiscoverRow] = Field(default_factory=list)
    warnings: list[BatchUploadWarning] = Field(default_factory=list)
    errors: list[BatchUploadWarning] = Field(default_factory=list)


class BatchUploadExecuteRequest(BaseModel):
    plan_id: str
    accepted_row_ids: list[str] = Field(default_factory=list)


class BatchUploadExecuteRowResult(BaseModel):
    row_id: str
    target_id: str
    action: Literal['create_new', 'append_existing', 'skip']
    accepted_images: int
    skipped_images: int = 0
    encounter_folder: str
    archive_paths_written: list[str] = Field(default_factory=list)
    warnings: list[BatchUploadWarning] = Field(default_factory=list)
    errors: list[BatchUploadWarning] = Field(default_factory=list)


class BatchUploadExecuteSummary(BaseModel):
    executed_rows: int
    created_ids: int
    appended_ids: int
    accepted_images: int
    skipped_images: int
    rows_with_errors: int


class BatchUploadExecuteResponse(BaseModel):
    status: Literal['ok', 'partial', 'error']
    plan_id: str
    batch_id: str
    target_archive: Literal['gallery', 'query']
    summary: BatchUploadExecuteSummary
    rows: list[BatchUploadExecuteRowResult] = Field(default_factory=list)
    message: str


class BatchUploadUploadResponse(BaseModel):
    upload_token: str
    file_count: int
    root_entries: list[str] = Field(default_factory=list)
