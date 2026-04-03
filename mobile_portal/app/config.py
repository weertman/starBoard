from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


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
    megastar_enabled: bool
    megastar_model_key_override: str | None
    megastar_artifact_root: Path
    megastar_registry_path: Path
    megastar_require_fresh_assets: bool


@dataclass(frozen=True)
class MegaStarCapabilityStatus:
    enabled: bool
    state: Literal['enabled', 'disabled', 'unavailable']
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
        archive_dir=archive_dir,
        host=os.getenv('STARBOARD_MOBILE_HOST', '127.0.0.1'),
        port=int(os.getenv('STARBOARD_MOBILE_PORT', '8091')),
        initial_image_window=int(os.getenv('STARBOARD_MOBILE_INITIAL_IMAGE_WINDOW', '4')),
        image_page_size=int(os.getenv('STARBOARD_MOBILE_IMAGE_PAGE_SIZE', '4')),
        max_upload_mb=int(os.getenv('STARBOARD_MOBILE_MAX_UPLOAD_MB', '250')),
        cf_bypass_localhost=os.getenv('STARBOARD_MOBILE_CF_BYPASS_LOCALHOST', '0') == '1',
        preview_cache_dir=cache_dir,
        megastar_enabled=os.getenv('STARBOARD_MOBILE_MEGASTAR_ENABLED', '0') == '1',
        megastar_model_key_override=megastar_model_key_override,
        megastar_artifact_root=megastar_artifact_root,
        megastar_registry_path=megastar_registry_path,
        megastar_require_fresh_assets=os.getenv('STARBOARD_MOBILE_MEGASTAR_REQUIRE_FRESH_ASSETS', '1') == '1',
    )


def get_megastar_capability_status(settings: Settings | None = None) -> MegaStarCapabilityStatus:
    settings = settings or get_settings()
    if not settings.megastar_enabled:
        return MegaStarCapabilityStatus(enabled=False, state='disabled', reason='feature_flag_disabled')

    if not settings.megastar_registry_path.exists():
        return MegaStarCapabilityStatus(enabled=False, state='unavailable', reason='registry_missing')

    try:
        registry = json.loads(settings.megastar_registry_path.read_text())
    except (OSError, json.JSONDecodeError):
        return MegaStarCapabilityStatus(enabled=False, state='unavailable', reason='registry_invalid')

    model_key = settings.megastar_model_key_override or registry.get('active_model')
    if not model_key:
        return MegaStarCapabilityStatus(enabled=False, state='unavailable', reason='model_not_configured')

    model_entry = registry.get('models', {}).get(model_key)
    if not isinstance(model_entry, dict):
        return MegaStarCapabilityStatus(enabled=False, state='unavailable', reason='model_not_registered', model_key=model_key)

    artifact_dir = settings.megastar_artifact_root / model_key
    required_paths = [
        artifact_dir / 'embeddings' / 'gallery_image_embeddings.npz',
        artifact_dir / 'embeddings' / 'gallery_image_paths.json',
        artifact_dir / 'similarity' / 'gallery_image_index.json',
    ]
    if any(not path.exists() for path in required_paths):
        return MegaStarCapabilityStatus(
            enabled=False,
            state='unavailable',
            reason='required_assets_missing',
            model_key=model_key,
            artifact_dir=artifact_dir,
        )

    pending_ids = registry.get('pending_ids', {})
    pending_gallery = pending_ids.get('gallery', []) if isinstance(pending_ids, dict) else []
    pending_queries = pending_ids.get('queries', []) if isinstance(pending_ids, dict) else []
    if settings.megastar_require_fresh_assets and (pending_gallery or pending_queries):
        return MegaStarCapabilityStatus(
            enabled=False,
            state='unavailable',
            reason='stale_artifacts',
            model_key=model_key,
            artifact_dir=artifact_dir,
        )

    return MegaStarCapabilityStatus(
        enabled=True,
        state='enabled',
        model_key=model_key,
        artifact_dir=artifact_dir,
    )
