from __future__ import annotations

from pathlib import Path

from src.data.ingest import detect_folder_depth

from ..adapters.batch_upload_adapter import discover_batch_source, generate_plan_id
from ..models.batch_upload_api import (
    BatchUploadDiscoverRequest,
    BatchUploadDiscoverResponse,
    BatchUploadDiscoverSummary,
)
from ..models.batch_upload_plan import PlannedBatch, put_plan


def _resolved_mode(source_path: Path, requested_mode: str) -> str:
    if requested_mode == 'auto':
        detected = detect_folder_depth(source_path)
        if detected == 'single_id':
            return 'encounters'
        if detected in {'flat', 'encounters', 'grouped', 'empty'}:
            return detected
        return 'empty'
    if requested_mode in {'flat', 'encounters', 'grouped'}:
        return requested_mode
    return 'empty'


def build_discover_preview(req: BatchUploadDiscoverRequest) -> BatchUploadDiscoverResponse:
    source_path = Path(req.import_source.path)
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
    detected_ids = {row.transformed_target_id for row in rows}
    existing_ids = sum(1 for row in rows if row.target_exists)
    new_ids = sum(1 for row in rows if not row.target_exists)
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
            new_ids=new_ids,
            existing_ids=existing_ids,
            warnings=warnings_count,
            errors=0,
        ),
        rows=rows,
        warnings=[],
        errors=[],
    )
