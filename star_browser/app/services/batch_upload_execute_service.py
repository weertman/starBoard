from __future__ import annotations

from datetime import date
from pathlib import Path

from src.data import archive_paths as ap
from src.data.batch_undo import generate_batch_id, record_batch_upload
from src.data.csv_io import append_row
from src.data.id_registry import id_exists
from src.data.ingest import ensure_encounter_name, place_images

from ..models.batch_upload_api import (
    BatchUploadExecuteRequest,
    BatchUploadExecuteResponse,
    BatchUploadExecuteRowResult,
    BatchUploadExecuteSummary,
    BatchUploadWarning,
)
from ..models.batch_upload_plan import PlannedBatch, PlannedBatchRow, get_plan


class BatchUploadPlanNotFoundError(Exception):
    pass


def _target_name(target_archive: str) -> str:
    return 'Gallery' if target_archive == 'gallery' else 'Queries'


def _encounter_name_for_row(plan: PlannedBatch, row: PlannedBatchRow) -> tuple[str, date | None]:
    if row.encounter_folder_name and row.encounter_date:
        obs_date = date.fromisoformat(row.encounter_date)
        return ensure_encounter_name(obs_date.year, obs_date.month, obs_date.day, row.encounter_suffix or ''), obs_date
    if plan.flat_encounter_date:
        obs_date = date.fromisoformat(plan.flat_encounter_date)
        return ensure_encounter_name(obs_date.year, obs_date.month, obs_date.day, plan.flat_encounter_suffix), obs_date
    today = date.today()
    return ensure_encounter_name(today.year, today.month, today.day, plan.flat_encounter_suffix), today


def _create_new_id_row(target_archive: str, id_str: str, plan: PlannedBatch) -> None:
    target = _target_name(target_archive)
    csv_path, header = ap.metadata_csv_for(target)
    id_col = ap.id_column_name(target)
    row = {col: '' for col in header}
    row[id_col] = id_str
    loc = plan.batch_location
    if loc.location:
        row['location'] = loc.location
    if loc.latitude:
        row['latitude'] = loc.latitude
    if loc.longitude:
        row['longitude'] = loc.longitude
    append_row(csv_path, header, row)


def execute_batch_upload(req: BatchUploadExecuteRequest) -> BatchUploadExecuteResponse:
    plan = get_plan(req.plan_id)
    if plan is None:
        raise BatchUploadPlanNotFoundError(req.plan_id)

    selected_rows = [row for row in plan.rows if row.row_id in set(req.accepted_row_ids)]
    target = _target_name(plan.target_archive)
    root = ap.root_for(target)
    batch_id = generate_batch_id()
    file_ops: list[tuple[Path, Path]] = []
    results: list[BatchUploadExecuteRowResult] = []
    created_ids: set[str] = set()
    appended_ids: set[str] = set()

    for row in selected_rows:
        encounter_name, obs_date = _encounter_name_for_row(plan, row)
        existed_before = id_exists(target, row.transformed_target_id)
        report = place_images(root, row.transformed_target_id, encounter_name, row.files, move=False, observation_date=obs_date)
        for op in report.ops:
            file_ops.append((op.src, op.dest))

        if not existed_before and report.ops:
            _create_new_id_row(plan.target_archive, row.transformed_target_id, plan)
            created_ids.add(row.transformed_target_id)
        elif report.ops:
            appended_ids.add(row.transformed_target_id)

        results.append(BatchUploadExecuteRowResult(
            row_id=row.row_id,
            target_id=row.transformed_target_id,
            action=row.action,  # type: ignore[arg-type]
            accepted_images=len(report.ops),
            skipped_images=max(0, len(row.files) - len(report.ops)),
            encounter_folder=encounter_name,
            archive_paths_written=[str(op.dest.relative_to(root)) for op in report.ops],
            warnings=[],
            errors=[BatchUploadWarning(code='transfer_error', message=e, row_id=row.row_id) for e in report.errors],
        ))

    if file_ops:
        record_batch_upload(target, batch_id, file_ops, created_ids, results[0].encounter_folder if results else 'batch_upload')

    rows_with_errors = sum(1 for row in results if row.errors)
    status = 'ok' if results and rows_with_errors == 0 else 'partial' if results else 'error'
    return BatchUploadExecuteResponse(
        status=status,  # type: ignore[arg-type]
        plan_id=req.plan_id,
        batch_id=batch_id,
        target_archive=plan.target_archive,  # type: ignore[arg-type]
        summary=BatchUploadExecuteSummary(
            executed_rows=len(results),
            created_ids=len(created_ids),
            appended_ids=len(appended_ids),
            accepted_images=sum(r.accepted_images for r in results),
            skipped_images=sum(r.skipped_images for r in results),
            rows_with_errors=rows_with_errors,
        ),
        rows=results,
        message='Batch upload completed.' if results else 'No rows executed.',
    )
