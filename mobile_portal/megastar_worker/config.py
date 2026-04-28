from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from mobile_portal.app.adapters.megastar_artifact_loader import load_megastar_artifact_availability
from mobile_portal.app.config import Settings as PortalSettings


@dataclass(frozen=True)
class WorkerSettings:
    repo_root: Path
    archive_dir: Path
    host: str
    port: int
    enabled: bool
    model_key_override: str | None
    artifact_root: Path
    registry_path: Path
    require_fresh_assets: bool
    max_upload_mb: int


def get_settings() -> WorkerSettings:
    repo_root = Path(__file__).resolve().parents[2]
    archive_dir = Path(os.getenv('STARBOARD_ARCHIVE_DIR', str(repo_root / 'archive'))).expanduser().resolve()
    artifact_root = Path(os.getenv('STARBOARD_MEGASTAR_ARTIFACT_ROOT', str(archive_dir / '_dl_precompute'))).expanduser().resolve()
    registry_path = Path(os.getenv('STARBOARD_MEGASTAR_REGISTRY_PATH', str(artifact_root / '_dl_registry.json'))).expanduser().resolve()
    return WorkerSettings(
        repo_root=repo_root,
        archive_dir=archive_dir,
        host=os.getenv('STARBOARD_MEGASTAR_WORKER_HOST', '127.0.0.1'),
        port=int(os.getenv('STARBOARD_MEGASTAR_WORKER_PORT', '8093')),
        enabled=os.getenv('STARBOARD_MEGASTAR_WORKER_ENABLED', '0') == '1',
        model_key_override=os.getenv('STARBOARD_MEGASTAR_MODEL_KEY') or None,
        artifact_root=artifact_root,
        registry_path=registry_path,
        require_fresh_assets=os.getenv('STARBOARD_MEGASTAR_REQUIRE_FRESH_ASSETS', '1') == '1',
        max_upload_mb=int(os.getenv('STARBOARD_MEGASTAR_MAX_UPLOAD_MB', '250')),
    )


def as_portal_settings(settings: WorkerSettings | None = None) -> PortalSettings:
    settings = settings or get_settings()
    return PortalSettings(
        repo_root=settings.repo_root,
        archive_dir=settings.archive_dir,
        host=settings.host,
        port=settings.port,
        initial_image_window=4,
        image_page_size=4,
        max_upload_mb=settings.max_upload_mb,
        cf_bypass_localhost=False,
        preview_cache_dir=(settings.repo_root / 'mobile_portal/.cache/previews').resolve(),
        megastar_enabled=settings.enabled,
        megastar_backend='local',
        megastar_worker_url=f'http://{settings.host}:{settings.port}',
        megastar_worker_timeout_seconds=5.0,
        megastar_model_key_override=settings.model_key_override,
        megastar_artifact_root=settings.artifact_root,
        megastar_registry_path=settings.registry_path,
        megastar_require_fresh_assets=settings.require_fresh_assets,
    )


def capability_status(settings: WorkerSettings | None = None):
    settings = settings or get_settings()
    if not settings.enabled:
        class Disabled:
            enabled = False
            state = 'disabled'
            reason = 'feature_flag_disabled'
            model_key = settings.model_key_override
            artifact_dir = None
        return Disabled()

    return load_megastar_artifact_availability(as_portal_settings(settings))
