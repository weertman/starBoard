from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .image_manifest_adapter import image_descriptor, list_entity_images


class MegaStarResultResolutionError(ValueError):
    pass


@dataclass(frozen=True)
class MegaStarArtifactMatch:
    entity_id: str
    local_idx: int
    artifact_path: str
    entity_type: str = 'gallery'


class MegaStarResultResolver:
    def resolve_best_match(self, match: MegaStarArtifactMatch) -> dict:
        if match.entity_type != 'gallery':
            raise MegaStarResultResolutionError(f'Unsupported entity type: {match.entity_type}')

        images = list_entity_images(match.entity_type, match.entity_id)
        if not images:
            raise MegaStarResultResolutionError(f'No archive images found for {match.entity_type}:{match.entity_id}')

        resolved_index = self._resolve_image_index(images, match)
        return image_descriptor(match.entity_type, match.entity_id, resolved_index, images[resolved_index])

    def _resolve_image_index(self, images: list[Path], match: MegaStarArtifactMatch) -> int:
        artifact_name = self._artifact_name(match.artifact_path)
        if artifact_name:
            by_name = [idx for idx, path in enumerate(images) if path.stem == artifact_name or path.name == artifact_name]
            if len(by_name) == 1:
                return by_name[0]

        if 0 <= match.local_idx < len(images):
            candidate = images[match.local_idx]
            if not artifact_name or candidate.stem == artifact_name or candidate.name == artifact_name:
                return match.local_idx

        if artifact_name:
            exact_name = [idx for idx, path in enumerate(images) if path.name == artifact_name]
            if len(exact_name) == 1:
                return exact_name[0]

        raise MegaStarResultResolutionError(
            f'Could not resolve portable archive image for gallery:{match.entity_id} local_idx={match.local_idx}'
        )

    def _artifact_name(self, artifact_path: str) -> str | None:
        portable = artifact_path.replace('\\', '/')
        name = Path(portable).name
        if not name:
            return None
        stem = Path(name).stem
        return stem or name
