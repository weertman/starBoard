from __future__ import annotations

from pathlib import Path

from fastapi import HTTPException, status

from src.data.image_formats import is_importable_image
from src.data.ingest import detect_folder_depth

from ..adapters.batch_upload_adapter import discover_batch_source, generate_plan_id
from ..models.batch_upload_api import (
    BatchUploadDiscoverRequest,
    BatchUploadDiscoverResponse,
    BatchUploadDiscoverSummary,
)
from ..models.batch_upload_plan import PlannedBatch, put_plan
from .batch_upload_upload_service import resolve_uploaded_bundle_path


def _resolved_mode(source_path: Path, requested_mode: str) -> str:
    if requested_mode == 'flat':
        return 'flat'
    detected = detect_folder_depth(source_path)
    if detected == 'ids':
        return 'encounters'
    if detected in {'single_id', 'flat', 'grouped', 'empty'}:
        return detected
    if requested_mode in {'encounters', 'grouped'}:
        return requested_mode
    return 'empty'


def _source_path_for_request(req: BatchUploadDiscoverRequest) -> Path:
    resolved = resolve_uploaded_bundle_path(req.import_source.upload_token)
    if resolved is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='upload_token_not_found')
    return resolved


def build_discover_preview(req: BatchUploadDiscoverRequest) -> BatchUploadDiscoverResponse:
    source_path = _source_path_for_request(req)
    resolved = _resolved_mode(source_path, req.discovery_mode)
    plan_id = generate_plan_id()
    planned_rows = discover_batch_source(
        source_path,
        requested_mode=req.discovery_mode,
        target_archive=req.target_archive,
        id_prefix=req.id_prefix,
        id_suffix=req.id_suffix,
    )
    put_plan(
        PlannedBatch(
            plan_id=plan_id,
            target_archive=req.target_archive,
            requested_discovery_mode=req.discovery_mode,
            resolved_discovery_mode=resolved,
            id_prefix=req.id_prefix,
            id_suffix=req.id_suffix,
            flat_encounter_date=req.flat_encounter_date,
            flat_encounter_suffix=req.flat_encounter_suffix,
            batch_location=req.batch_location,
            rows=planned_rows,
        )
    )
    rows = [row.to_api_row() for row in planned_rows]
    new_id_values = {row.transformed_target_id for row in rows if not row.target_exists}
    existing_id_values = {row.transformed_target_id for row in rows if row.target_exists}
    detected_ids = new_id_values | existing_id_values
    warnings_count = sum(len(row.warnings) for row in rows)
    return BatchUploadDiscoverResponse(
        plan_id=plan_id,
        target_archive=req.target_archive,
        requested_discovery_mode=req.discovery_mode,
        resolved_discovery_mode=resolved,
        summary=BatchUploadDiscoverSummary(
            detected_rows=len(rows),
            detected_ids=len(detected_ids),
            total_images=sum(row.image_count for row in rows),
            new_ids=len(new_id_values),
            existing_ids=len(existing_id_values),
            warnings=warnings_count,
            errors=0,
        ),
        rows=rows,
        warnings=[],
        errors=[],
    )
