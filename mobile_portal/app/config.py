from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class Settings:
    repo_root: Path
    archive_dir: Path
    host: str
    port: int
    initial_image_window: int
    image_page_size: int
    max_upload_mb: int
    cf_bypass_localhost: bool
    preview_cache_dir: Path
    megastar_enabled: bool
    megastar_backend: Literal['local', 'worker']
    megastar_worker_url: str
    megastar_worker_timeout_seconds: float
    megastar_model_key_override: str | None
    megastar_artifact_root: Path
    megastar_registry_path: Path
    megastar_require_fresh_assets: bool


@dataclass(frozen=True)
class MegaStarCapabilityStatus:
    enabled: bool
    state: Literal['enabled', 'disabled', 'unavailable']
    backend: Literal['local', 'worker']
    reason: str | None = None
    model_key: str | None = None
    artifact_dir: Path | None = None


def get_settings() -> Settings:
    repo_root = Path(__file__).resolve().parents[2]
    archive_dir = Path(os.getenv('STARBOARD_ARCHIVE_DIR', str(repo_root / 'archive'))).expanduser().resolve()
    cache_dir = Path(os.getenv('STARBOARD_MOBILE_PREVIEW_CACHE_DIR', str(repo_root / 'mobile_portal/.cache/previews'))).expanduser().resolve()
    megastar_artifact_root = Path(
        os.getenv('STARBOARD_MOBILE_MEGASTAR_ARTIFACT_ROOT', str(archive_dir / '_dl_precompute'))
    ).expanduser().resolve()
    megastar_registry_path = Path(
        os.getenv('STARBOARD_MOBILE_MEGASTAR_REGISTRY_PATH', str(megastar_artifact_root / '_dl_registry.json'))
    ).expanduser().resolve()
    megastar_model_key_override = os.getenv('STARBOARD_MOBILE_MEGASTAR_MODEL_KEY') or None
    return Settings(
        repo_root=repo_root,
        archive_dir=archive_dir,
        host=os.getenv('STARBOARD_MOBILE_HOST', '127.0.0.1'),
        port=int(os.getenv('STARBOARD_MOBILE_PORT', '8091')),
        initial_image_window=int(os.getenv('STARBOARD_MOBILE_INITIAL_IMAGE_WINDOW', '4')),
        image_page_size=int(os.getenv('STARBOARD_MOBILE_IMAGE_PAGE_SIZE', '4')),
        max_upload_mb=int(os.getenv('STARBOARD_MOBILE_MAX_UPLOAD_MB', '250')),
        cf_bypass_localhost=os.getenv('STARBOARD_MOBILE_CF_BYPASS_LOCALHOST', '0') == '1',
        preview_cache_dir=cache_dir,
        megastar_enabled=os.getenv('STARBOARD_MOBILE_MEGASTAR_ENABLED', '0') == '1',
        megastar_backend='worker' if os.getenv('STARBOARD_MOBILE_MEGASTAR_BACKEND', 'local').strip().lower() == 'worker' else 'local',
        megastar_worker_url=os.getenv('STARBOARD_MOBILE_MEGASTAR_WORKER_URL', 'http://127.0.0.1:8093').rstrip('/'),
        megastar_worker_timeout_seconds=float(os.getenv('STARBOARD_MOBILE_MEGASTAR_WORKER_TIMEOUT_SECONDS', '5')),
        megastar_model_key_override=megastar_model_key_override,
        megastar_artifact_root=megastar_artifact_root,
        megastar_registry_path=megastar_registry_path,
        megastar_require_fresh_assets=os.getenv('STARBOARD_MOBILE_MEGASTAR_REQUIRE_FRESH_ASSETS', '1') == '1',
    )


def get_megastar_capability_status(settings: Settings | None = None) -> MegaStarCapabilityStatus:
    from .services.megastar_backend_selector import get_megastar_capability_status as _get_megastar_capability_status

    return _get_megastar_capability_status(settings or get_settings())
