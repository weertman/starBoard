from __future__ import annotations

from pathlib import Path

from src.data.archive_paths import archive_root as sb_archive_root, root_for, metadata_csv_for, id_column_name, metadata_csv_paths_for_read
from src.data.id_registry import id_exists, list_ids
from src.data.csv_io import read_rows_multi, last_row_per_id


def get_archive_root() -> Path:
    return sb_archive_root()


def target_roots(entity_type: str) -> tuple[Path, Path, str]:
    canonical = 'Gallery' if entity_type == 'gallery' else 'Queries'
    root = root_for(canonical)
    csv_path, header = metadata_csv_for(canonical)
    return root, csv_path, id_column_name(canonical)


def entity_exists(entity_type: str, entity_id: str) -> bool:
    canonical = 'Gallery' if entity_type == 'gallery' else 'Queries'
    return id_exists(canonical, entity_id)


def list_entity_ids(entity_type: str) -> list[str]:
    canonical = 'Gallery' if entity_type == 'gallery' else 'Queries'
    exclude_silent = entity_type == 'query'
    return list_ids(canonical, exclude_silent=exclude_silent)


def latest_metadata_row(entity_type: str, entity_id: str) -> dict[str, str]:
    canonical = 'Gallery' if entity_type == 'gallery' else 'Queries'
    rows = read_rows_multi(metadata_csv_paths_for_read(canonical))
    row_map = last_row_per_id(rows, id_column_name(canonical))
    return dict(row_map.get(entity_id, {}))
