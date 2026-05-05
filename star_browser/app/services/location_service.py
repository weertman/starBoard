from __future__ import annotations

from src.data.location_sites import list_location_site_names, list_location_sites


def get_location_site_names() -> list[str]:
    return list_location_site_names()


def get_location_sites() -> dict:
    return {'sites': list_location_sites()}
