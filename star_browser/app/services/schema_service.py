from __future__ import annotations

from ..adapters.starboard_schema import project_schema


def get_metadata_schema() -> dict:
    return {'fields': project_schema()}
