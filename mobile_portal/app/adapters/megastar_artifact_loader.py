from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from src.dl import DL_AVAILABLE


@dataclass(frozen=True)
class MegaStarArtifactAvailability:
    enabled: bool
    state: Literal['enabled', 'disabled', 'unavailable']
    reason: str | None = None
    model_key: str | None = None
    artifact_dir: Path | None = None
    registry_path: Path | None = None
    checkpoint_path: Path | None = None
    gallery_embeddings_path: Path | None = None
    gallery_paths_path: Path | None = None
    gallery_index_path: Path | None = None
    metadata_path: Path | None = None
    embedding_dim: int | None = None
    image_size: int | None = None
    use_tta: bool | None = None
    raw_registry: dict[str, Any] | None = None
    raw_model_entry: dict[str, Any] | None = None
    raw_metadata: dict[str, Any] | None = None


def _resolve_repo_path(repo_root: Path, raw_path: str | None) -> Path | None:
    if not raw_path:
        return None
    candidate = Path(raw_path).expanduser()
    if candidate.exists():
        return candidate.resolve()
    portable = Path(raw_path.replace('\\', '/')).expanduser()
    if portable.exists():
        return portable.resolve()
    if portable.is_absolute():
        return portable
    rebased = (repo_root / portable).resolve()
    if rebased.exists():
        return rebased
    return rebased


def load_megastar_artifact_availability(settings) -> MegaStarArtifactAvailability:
    if not settings.megastar_enabled:
        return MegaStarArtifactAvailability(enabled=False, state='disabled', reason='feature_flag_disabled')

    registry_path = settings.megastar_registry_path
    if not registry_path.exists():
        return MegaStarArtifactAvailability(
            enabled=False,
            state='unavailable',
            reason='registry_missing',
            registry_path=registry_path,
        )

    try:
        registry = json.loads(registry_path.read_text())
    except (OSError, json.JSONDecodeError):
        return MegaStarArtifactAvailability(
            enabled=False,
            state='unavailable',
            reason='registry_invalid',
            registry_path=registry_path,
        )

    model_key = settings.megastar_model_key_override or registry.get('active_model')
    if not model_key:
        return MegaStarArtifactAvailability(
            enabled=False,
            state='unavailable',
            reason='model_not_configured',
            registry_path=registry_path,
            raw_registry=registry,
        )

    models = registry.get('models', {})
    model_entry = models.get(model_key)
    if not isinstance(model_entry, dict):
        return MegaStarArtifactAvailability(
            enabled=False,
            state='unavailable',
            reason='model_not_registered',
            model_key=model_key,
            registry_path=registry_path,
            raw_registry=registry,
        )

    if not model_entry.get('precomputed'):
        return MegaStarArtifactAvailability(
            enabled=False,
            state='unavailable',
            reason='model_not_precomputed',
            model_key=model_key,
            registry_path=registry_path,
            raw_registry=registry,
            raw_model_entry=model_entry,
        )

    artifact_dir = settings.megastar_artifact_root / model_key
    gallery_embeddings_path = artifact_dir / 'embeddings' / 'gallery_image_embeddings.npz'
    gallery_paths_path = artifact_dir / 'embeddings' / 'gallery_image_paths.json'
    gallery_index_path = artifact_dir / 'similarity' / 'gallery_image_index.json'
    metadata_path = artifact_dir / 'similarity' / 'metadata.json'
    required_paths = (gallery_embeddings_path, gallery_paths_path, gallery_index_path, metadata_path)
    if any(not path.exists() for path in required_paths):
        return MegaStarArtifactAvailability(
            enabled=False,
            state='unavailable',
            reason='required_assets_missing',
            model_key=model_key,
            artifact_dir=artifact_dir,
            registry_path=registry_path,
            raw_registry=registry,
            raw_model_entry=model_entry,
        )

    checkpoint_path = _resolve_repo_path(settings.repo_root, model_entry.get('checkpoint_path'))
    if checkpoint_path is None or not checkpoint_path.exists():
        return MegaStarArtifactAvailability(
            enabled=False,
            state='unavailable',
            reason='checkpoint_missing',
            model_key=model_key,
            artifact_dir=artifact_dir,
            registry_path=registry_path,
            checkpoint_path=checkpoint_path,
            raw_registry=registry,
            raw_model_entry=model_entry,
        )

    if not DL_AVAILABLE:
        return MegaStarArtifactAvailability(
            enabled=False,
            state='unavailable',
            reason='dl_unavailable',
            model_key=model_key,
            artifact_dir=artifact_dir,
            registry_path=registry_path,
            checkpoint_path=checkpoint_path,
            raw_registry=registry,
            raw_model_entry=model_entry,
        )

    try:
        metadata = json.loads(metadata_path.read_text())
    except (OSError, json.JSONDecodeError):
        return MegaStarArtifactAvailability(
            enabled=False,
            state='unavailable',
            reason='artifact_metadata_invalid',
            model_key=model_key,
            artifact_dir=artifact_dir,
            registry_path=registry_path,
            checkpoint_path=checkpoint_path,
            raw_registry=registry,
            raw_model_entry=model_entry,
        )

    use_tta = metadata.get('use_tta')
    image_size = metadata.get('image_size')
    embedding_dim = metadata.get('embedding_dim')
    if not isinstance(use_tta, bool) or not isinstance(image_size, int) or image_size <= 0 or not isinstance(embedding_dim, int) or embedding_dim <= 0:
        return MegaStarArtifactAvailability(
            enabled=False,
            state='unavailable',
            reason='artifact_metadata_incomplete',
            model_key=model_key,
            artifact_dir=artifact_dir,
            registry_path=registry_path,
            checkpoint_path=checkpoint_path,
            raw_registry=registry,
            raw_model_entry=model_entry,
            raw_metadata=metadata,
        )

    try:
        gallery_index = json.loads(gallery_index_path.read_text())
        gallery_paths = json.loads(gallery_paths_path.read_text())
    except (OSError, json.JSONDecodeError):
        return MegaStarArtifactAvailability(
            enabled=False,
            state='unavailable',
            reason='artifact_index_invalid',
            model_key=model_key,
            artifact_dir=artifact_dir,
            registry_path=registry_path,
            checkpoint_path=checkpoint_path,
            raw_registry=registry,
            raw_model_entry=model_entry,
            raw_metadata=metadata,
        )

    if not isinstance(gallery_index, list) or not isinstance(gallery_paths, dict):
        return MegaStarArtifactAvailability(
            enabled=False,
            state='unavailable',
            reason='artifact_index_invalid',
            model_key=model_key,
            artifact_dir=artifact_dir,
            registry_path=registry_path,
            checkpoint_path=checkpoint_path,
            raw_registry=registry,
            raw_model_entry=model_entry,
            raw_metadata=metadata,
        )

    for item in gallery_index[: min(len(gallery_index), 10)]:
        if not isinstance(item, dict):
            return MegaStarArtifactAvailability(
                enabled=False,
                state='unavailable',
                reason='artifact_index_invalid',
                model_key=model_key,
                artifact_dir=artifact_dir,
                registry_path=registry_path,
                checkpoint_path=checkpoint_path,
                raw_registry=registry,
                raw_model_entry=model_entry,
                raw_metadata=metadata,
            )
        if not isinstance(item.get('id'), str) or not isinstance(item.get('local_idx'), int) or not isinstance(item.get('path'), str):
            return MegaStarArtifactAvailability(
                enabled=False,
                state='unavailable',
                reason='artifact_index_invalid',
                model_key=model_key,
                artifact_dir=artifact_dir,
                registry_path=registry_path,
                checkpoint_path=checkpoint_path,
                raw_registry=registry,
                raw_model_entry=model_entry,
                raw_metadata=metadata,
            )

    pending_ids = registry.get('pending_ids', {}) if isinstance(registry.get('pending_ids', {}), dict) else {}
    pending_gallery = pending_ids.get('gallery', []) or []
    pending_queries = pending_ids.get('queries', []) or []
    # Pending IDs are logged but no longer block lookups — the existing
    # gallery embeddings are still valid for the IDs that have been computed.
    if pending_gallery or pending_queries:
        import logging
        _log = logging.getLogger('starBoard.megastar.artifacts')
        _log.warning(
            "MegaStar has pending IDs (gallery=%d, queries=%d) — "
            "results will not include these until precompute is re-run.",
            len(pending_gallery), len(pending_queries),
        )

    return MegaStarArtifactAvailability(
        enabled=True,
        state='enabled',
        model_key=model_key,
        artifact_dir=artifact_dir,
        registry_path=registry_path,
        checkpoint_path=checkpoint_path,
        gallery_embeddings_path=gallery_embeddings_path,
        gallery_paths_path=gallery_paths_path,
        gallery_index_path=gallery_index_path,
        metadata_path=metadata_path,
        embedding_dim=embedding_dim,
        image_size=image_size,
        use_tta=use_tta,
        raw_registry=registry,
        raw_model_entry=model_entry,
        raw_metadata=metadata,
    )
