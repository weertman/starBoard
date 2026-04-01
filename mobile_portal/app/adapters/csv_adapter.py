from __future__ import annotations

from src.data.archive_paths import metadata_csv_for
from src.data.csv_io import append_row


def write_metadata_row(entity_type: str, entity_id: str, metadata: dict[str, str]) -> str:
    canonical = 'Gallery' if entity_type == 'gallery' else 'Queries'
    csv_path, header = metadata_csv_for(canonical)
    row = dict(metadata)
    id_col = 'gallery_id' if entity_type == 'gallery' else 'query_id'
    row[id_col] = entity_id
    append_row(csv_path, header, row)
    return str(csv_path)
