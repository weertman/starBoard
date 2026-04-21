from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .batch_upload_api import BatchUploadDiscoverRow, BatchUploadLocationDraft


@dataclass
class PlannedBatchRow:
    row_id: str
    original_detected_id: str
    transformed_target_id: str
    action: str
    target_exists: bool
    files: list[Path] = field(default_factory=list)
    group_name: str | None = None
    encounter_folder_name: str | None = None
    encounter_date: str | None = None
    encounter_suffix: str | None = None

    @property
    def image_count(self) -> int:
        return len(self.files)

    def to_api_row(self) -> BatchUploadDiscoverRow:
        return BatchUploadDiscoverRow(
            row_id=self.row_id,
            original_detected_id=self.original_detected_id,
            transformed_target_id=self.transformed_target_id,
            action=self.action,  # type: ignore[arg-type]
            target_exists=self.target_exists,
            group_name=self.group_name,
            encounter_folder_name=self.encounter_folder_name,
            encounter_date=self.encounter_date,
            encounter_suffix=self.encounter_suffix,
            image_count=len(self.files),
            sample_labels=[p.name for p in self.files[:3]],
            source_ref=self._source_ref(),
            warnings=[],
        )

    def _source_ref(self) -> str:
        if self.group_name and self.encounter_folder_name:
            return f'{self.group_name}/{self.original_detected_id}/{self.encounter_folder_name}'
        if self.encounter_folder_name:
            return f'{self.original_detected_id}/{self.encounter_folder_name}'
        return self.original_detected_id


@dataclass
class PlannedBatch:
    plan_id: str
    target_archive: str
    requested_discovery_mode: str
    resolved_discovery_mode: str
    id_prefix: str
    id_suffix: str
    flat_encounter_date: str
    flat_encounter_suffix: str
    batch_location: BatchUploadLocationDraft
    rows: list[PlannedBatchRow] = field(default_factory=list)


_PLAN_STORE: dict[str, PlannedBatch] = {}


def put_plan(plan: PlannedBatch) -> None:
    _PLAN_STORE[plan.plan_id] = plan


def get_plan(plan_id: str) -> PlannedBatch | None:
    return _PLAN_STORE.get(plan_id)
