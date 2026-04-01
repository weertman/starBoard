from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    archive_dir: Path
    host: str
    port: int
    initial_image_window: int
    image_page_size: int
    max_upload_mb: int
    cf_bypass_localhost: bool
    preview_cache_dir: Path


def get_settings() -> Settings:
    repo_root = Path(__file__).resolve().parents[2]
    archive_dir = Path(os.getenv('STARBOARD_ARCHIVE_DIR', str(repo_root / 'archive'))).expanduser().resolve()
    cache_dir = Path(os.getenv('STARBOARD_MOBILE_PREVIEW_CACHE_DIR', str(repo_root / 'mobile_portal/.cache/previews'))).expanduser().resolve()
    return Settings(
        archive_dir=archive_dir,
        host=os.getenv('STARBOARD_MOBILE_HOST', '127.0.0.1'),
        port=int(os.getenv('STARBOARD_MOBILE_PORT', '8091')),
        initial_image_window=int(os.getenv('STARBOARD_MOBILE_INITIAL_IMAGE_WINDOW', '4')),
        image_page_size=int(os.getenv('STARBOARD_MOBILE_IMAGE_PAGE_SIZE', '4')),
        max_upload_mb=int(os.getenv('STARBOARD_MOBILE_MAX_UPLOAD_MB', '250')),
        cf_bypass_localhost=os.getenv('STARBOARD_MOBILE_CF_BYPASS_LOCALHOST', '0') == '1',
        preview_cache_dir=cache_dir,
    )
