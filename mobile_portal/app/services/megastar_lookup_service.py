from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from time import perf_counter

import numpy as np
from fastapi import HTTPException, status

from ..adapters.megastar_artifact_loader import MegaStarArtifactAvailability, load_megastar_artifact_availability
from ..adapters.megastar_model_adapter import MegaStarModelAdapter, ReIDAdapter
from ..adapters.megastar_query_preprocess import MegaStarQueryPreprocessError, MegaStarQueryPreprocessor
from ..adapters.megastar_result_resolver import MegaStarArtifactMatch, MegaStarResultResolutionError, MegaStarResultResolver
from ..config import Settings, get_settings
from ..models.megastar_api import MegaStarLookupCandidate, MegaStarLookupResponse
from src.data.encounter_info import get_encounter_date

ALLOWED_IMAGE_CONTENT_TYPES = {
    'image/jpeg',
    'image/jpg',
    'image/png',
    'image/bmp',
    'image/tiff',
    'image/webp',
}
TOP_IMAGE_MATCHES = 25
MAX_CANDIDATES = 5
MIN_CANDIDATE_SCORE = 0.05
WEAK_CANDIDATE_SCORE = 0.35
WEAK_MARGIN_SCORE = 0.03


class MegaStarLookupUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class MegaStarImageSearchHit:
    index: int
    entity_id: str
    local_idx: int
    artifact_path: str
    retrieval_score: float


@dataclass(frozen=True)
class MegaStarIdCandidate:
    entity_id: str
    retrieval_score: float
    best_hit: MegaStarImageSearchHit
    encounter: str | None
    encounter_date: str | None


@dataclass(frozen=True)
class MegaStarGallerySearchIndex:
    gallery_matrix: np.ndarray
    image_index: tuple[dict, ...]


class MegaStarLookupService:
    def __init__(
        self,
        *,
        settings: Settings | None = None,
        backend_factory: type[ReIDAdapter] = ReIDAdapter,
        yolo_preprocessor=None,
        result_resolver: MegaStarResultResolver | None = None,
    ):
        self.settings = settings or get_settings()
        self.backend_factory = backend_factory
        self.yolo_preprocessor = yolo_preprocessor
        self.result_resolver = result_resolver or MegaStarResultResolver()

    def lookup_upload(self, *, filename: str, content: bytes, content_type: str | None = None) -> MegaStarLookupResponse:
        started = perf_counter()
        self._validate_upload(filename=filename, content=content, content_type=content_type)
        availability = load_megastar_artifact_availability(self.settings)
        if not availability.enabled:
            raise MegaStarLookupUnavailable(availability.reason or 'megastar_unavailable')

        try:
            preprocessor = MegaStarQueryPreprocessor(
                image_size=availability.image_size or 384,
                yolo_preprocessor=self.yolo_preprocessor,
            )
            preprocessed = preprocessor.preprocess_upload_bytes(content)
            model = MegaStarModelAdapter(availability=availability, backend_factory=self.backend_factory)
            query_embedding = model.extract_embedding(preprocessed.image_tensor)
            hits = self.search_image_hits(query_embedding=query_embedding, availability=availability)
            candidates = self.aggregate_id_candidates(hits)
            response_candidates = self._build_response_candidates(candidates)
        except MegaStarQueryPreprocessError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        except MegaStarResultResolutionError as exc:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail='result_resolution_failed') from exc
        except MegaStarLookupUnavailable:
            raise
        except Exception as exc:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail='megastar_lookup_failed') from exc

        status_value = self._result_status(candidates)
        return MegaStarLookupResponse(
            query_image_name=filename,
            status=status_value,
            candidates=response_candidates,
            processing_ms=int((perf_counter() - started) * 1000),
            capability_state=availability.state,
            availability_reason=None,
        )

    def search_image_hits(
        self,
        query_embedding: np.ndarray,
        *,
        availability: MegaStarArtifactAvailability,
        limit: int = TOP_IMAGE_MATCHES,
    ) -> list[MegaStarImageSearchHit]:
        gallery = self._load_gallery_search_index(availability)
        query_vector = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        if gallery.gallery_matrix.shape[1] != query_vector.shape[0]:
            raise MegaStarLookupUnavailable('embedding_dim_mismatch')

        norm = float(np.linalg.norm(query_vector))
        if not np.isfinite(norm) or norm <= 0:
            raise MegaStarLookupUnavailable('invalid_query_embedding')
        query_vector = query_vector / norm

        similarity = gallery.gallery_matrix @ query_vector
        if similarity.size == 0:
            return []

        top_n = min(limit, similarity.size)
        ranked = np.argsort(similarity)[::-1][:top_n]
        hits: list[MegaStarImageSearchHit] = []
        for rank_idx in ranked.tolist():
            score = float(similarity[rank_idx])
            if not np.isfinite(score):
                continue
            item = gallery.image_index[rank_idx]
            hits.append(
                MegaStarImageSearchHit(
                    index=rank_idx,
                    entity_id=item['id'],
                    local_idx=item['local_idx'],
                    artifact_path=item['path'],
                    retrieval_score=score,
                )
            )
        return hits

    def aggregate_id_candidates(self, hits: list[MegaStarImageSearchHit], *, max_candidates: int = MAX_CANDIDATES) -> list[MegaStarIdCandidate]:
        grouped: dict[str, MegaStarImageSearchHit] = {}
        for hit in hits:
            current = grouped.get(hit.entity_id)
            if current is None or hit.retrieval_score > current.retrieval_score:
                grouped[hit.entity_id] = hit

        ranked = sorted(grouped.values(), key=lambda hit: (-hit.retrieval_score, hit.entity_id))
        filtered: list[MegaStarIdCandidate] = []
        for hit in ranked:
            if hit.retrieval_score < MIN_CANDIDATE_SCORE:
                continue
            encounter = None
            encounter_date = None
            try:
                descriptor = self.result_resolver.resolve_best_match(
                    MegaStarArtifactMatch(entity_id=hit.entity_id, local_idx=hit.local_idx, artifact_path=hit.artifact_path)
                )
                encounter = descriptor.get('encounter') or None
                encounter_date = self._encounter_date_for(hit.entity_id, encounter)
            except MegaStarResultResolutionError:
                encounter = None
                encounter_date = None
            filtered.append(
                MegaStarIdCandidate(
                    entity_id=hit.entity_id,
                    retrieval_score=hit.retrieval_score,
                    best_hit=hit,
                    encounter=encounter,
                    encounter_date=encounter_date,
                )
            )
            if len(filtered) >= max_candidates:
                break
        return filtered

    def _build_response_candidates(self, candidates: list[MegaStarIdCandidate]) -> list[MegaStarLookupCandidate]:
        payload: list[MegaStarLookupCandidate] = []
        for idx, candidate in enumerate(candidates, start=1):
            best_match_image = self.result_resolver.resolve_best_match(
                MegaStarArtifactMatch(
                    entity_id=candidate.best_hit.entity_id,
                    local_idx=candidate.best_hit.local_idx,
                    artifact_path=candidate.best_hit.artifact_path,
                )
            )
            payload.append(
                MegaStarLookupCandidate(
                    rank=idx,
                    entity_id=candidate.entity_id,
                    retrieval_score=round(candidate.retrieval_score, 6),
                    best_match_image=best_match_image,
                    best_match_label=best_match_image.get('label'),
                    encounter=candidate.encounter,
                    encounter_date=candidate.encounter_date,
                )
            )
        return payload

    def _result_status(self, candidates: list[MegaStarIdCandidate]) -> str:
        if not candidates:
            return 'empty'
        top_score = candidates[0].retrieval_score
        margin = top_score - candidates[1].retrieval_score if len(candidates) > 1 else top_score
        if top_score < WEAK_CANDIDATE_SCORE or margin < WEAK_MARGIN_SCORE:
            return 'weak'
        return 'ok'

    def _encounter_date_for(self, entity_id: str, encounter: str | None) -> str | None:
        if not encounter:
            return None
        date_value = get_encounter_date('Gallery', entity_id, encounter)
        if date_value:
            return date_value.isoformat()
        if encounter.startswith('enc-') and len(encounter) >= 14:
            maybe_date = encounter[4:14]
            if len(maybe_date) == 10 and maybe_date[4] == '-' and maybe_date[7] == '-':
                return maybe_date
        return None

    def _load_gallery_search_index(self, availability: MegaStarArtifactAvailability) -> MegaStarGallerySearchIndex:
        if availability.gallery_embeddings_path is None or availability.gallery_index_path is None:
            raise MegaStarLookupUnavailable('gallery_artifacts_missing')
        return _cached_gallery_search_index(
            str(availability.gallery_embeddings_path),
            str(availability.gallery_index_path),
            availability.embedding_dim or 0,
        )

    def _validate_upload(self, *, filename: str, content: bytes, content_type: str | None) -> None:
        if not filename:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='filename_required')
        if not content:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='empty_upload')
        if len(content) > self.settings.max_upload_mb * 1024 * 1024:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='file_too_large')
        if content_type and content_type.lower() not in ALLOWED_IMAGE_CONTENT_TYPES:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='unsupported_media_type')


@lru_cache(maxsize=4)
def _cached_gallery_search_index(
    gallery_embeddings_path: str,
    gallery_index_path: str,
    expected_embedding_dim: int,
) -> MegaStarGallerySearchIndex:
    with np.load(gallery_embeddings_path) as embedding_store:
        embeddings_by_id = {key: np.asarray(embedding_store[key], dtype=np.float32) for key in embedding_store.files}

    with open(gallery_index_path, 'r', encoding='utf-8') as handle:
        image_index = json.load(handle)

    if not isinstance(image_index, list) or not image_index:
        raise MegaStarLookupUnavailable('gallery_index_empty')

    flattened: list[np.ndarray] = []
    normalized_index: list[dict] = []
    for item in image_index:
        if not isinstance(item, dict):
            raise MegaStarLookupUnavailable('gallery_index_invalid')
        entity_id = item.get('id')
        local_idx = item.get('local_idx')
        artifact_path = item.get('path')
        if not isinstance(entity_id, str) or not isinstance(local_idx, int) or not isinstance(artifact_path, str):
            raise MegaStarLookupUnavailable('gallery_index_invalid')
        entity_embeddings = embeddings_by_id.get(entity_id)
        if entity_embeddings is None or local_idx < 0 or local_idx >= len(entity_embeddings):
            raise MegaStarLookupUnavailable('gallery_index_mismatch')
        embedding = np.asarray(entity_embeddings[local_idx], dtype=np.float32).reshape(-1)
        if expected_embedding_dim and embedding.shape[0] != expected_embedding_dim:
            raise MegaStarLookupUnavailable('embedding_dim_mismatch')
        norm = float(np.linalg.norm(embedding))
        if not np.isfinite(norm) or norm <= 0:
            raise MegaStarLookupUnavailable('gallery_embedding_invalid')
        flattened.append((embedding / norm).astype(np.float32))
        normalized_index.append({'id': entity_id, 'local_idx': local_idx, 'path': artifact_path})

    return MegaStarGallerySearchIndex(gallery_matrix=np.stack(flattened), image_index=tuple(normalized_index))


@lru_cache(maxsize=1)
def get_megastar_lookup_service() -> MegaStarLookupService:
    return MegaStarLookupService()
