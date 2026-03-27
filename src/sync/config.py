"""
Sync configuration for starBoard.

Resolves the lab identity for this machine. Used to auto-populate
the sync metadata fields (source_lab, modified_by_lab) in CSV rows.

Resolution order:
  1. STARBOARD_LAB_ID environment variable
  2. archive/starboard_sync_config.json  {"lab_id": "..."}
  3. Machine hostname (fallback)
"""
from __future__ import annotations

import json
import logging
import os
import socket
from pathlib import Path
from typing import Optional

log = logging.getLogger("starBoard.sync.config")

_cached_lab_id: Optional[str] = None


def _config_file_path() -> Path:
    """Path to the optional sync config file inside the archive."""
    # Lazy import to avoid circular dependency with archive_paths
    try:
        from src.data.archive_paths import archive_root
        return archive_root() / "starboard_sync_config.json"
    except Exception:
        return Path("archive") / "starboard_sync_config.json"


def get_lab_id() -> str:
    """
    Return the lab/machine identifier for this starBoard instance.

    Resolution order:
      1. STARBOARD_LAB_ID env var
      2. archive/starboard_sync_config.json  {"lab_id": "..."}
      3. Machine hostname
    """
    global _cached_lab_id
    if _cached_lab_id is not None:
        return _cached_lab_id

    # 1. Environment variable
    env = os.getenv("STARBOARD_LAB_ID", "").strip()
    if env:
        _cached_lab_id = env
        log.info("Lab ID from env: %s", env)
        return env

    # 2. Config file
    cfg_path = _config_file_path()
    if cfg_path.exists():
        try:
            with cfg_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            lab = data.get("lab_id", "").strip()
            if lab:
                _cached_lab_id = lab
                log.info("Lab ID from config: %s", lab)
                return lab
        except Exception as e:
            log.warning("Failed to read sync config %s: %s", cfg_path, e)

    # 3. Hostname fallback
    hostname = socket.gethostname().strip() or "unknown"
    _cached_lab_id = hostname
    log.info("Lab ID from hostname: %s", hostname)
    return hostname


def set_lab_id(lab_id: str) -> None:
    """
    Persist lab_id to the sync config file and update the cache.
    """
    global _cached_lab_id
    _cached_lab_id = lab_id

    cfg_path = _config_file_path()
    cfg_path.parent.mkdir(parents=True, exist_ok=True)

    # Read existing config if present, merge
    data = {}
    if cfg_path.exists():
        try:
            with cfg_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            pass

    data["lab_id"] = lab_id

    with cfg_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    log.info("Saved lab ID to %s: %s", cfg_path, lab_id)


def invalidate_lab_id_cache() -> None:
    """Clear the cached lab ID (e.g. after config change)."""
    global _cached_lab_id
    _cached_lab_id = None
