from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    repo_root: Path
    archive_dir: Path
    host: str
    port: int
    staging_dir: Path


def get_settings() -> Settings:
    repo_root = Path(__file__).resolve().parents[2]
    archive_dir = Path(os.getenv('STARBOARD_ARCHIVE_DIR', str(repo_root / 'archive'))).expanduser().resolve()
    staging_dir = Path(os.getenv('STAR_BROWSER_STAGING_DIR', str(repo_root / 'star_browser/.staging'))).expanduser().resolve()
    return Settings(
        repo_root=repo_root,
        archive_dir=archive_dir,
        host=os.getenv('STAR_BROWSER_HOST', '127.0.0.1'),
        port=int(os.getenv('STAR_BROWSER_PORT', '8094')),
        staging_dir=staging_dir,
    )
