# src/search/fields_text_embed_bge.py
from __future__ import annotations
from typing import Dict, Any, Tuple
import logging
import numpy as np

from .interfaces import Row
from .embed_store import (
    DEFAULT_MODEL_ID,
    load_vectors_for_field,
    upsert_single_query_vector,
)

_log = logging.getLogger("starBoard.search.field.text_embed")

class TextEmbeddingBGEScorer:
    """
    BGE‑M3 sentence embeddings + cosine similarity mapped to [0,1].
    - Gallery vectors are loaded from archive/.../metadata_embeddings.json (does not need the model).
    - Query vectors are embedded on demand and cached; if embedding fails, we skip this field.
    """
    def __init__(self, field: str, model_id: str = DEFAULT_MODEL_ID):
        self.name = field
        self.model_id = model_id
        self.vecs_by_id: Dict[str, np.ndarray] = {}
        self._fatal_disabled = False     # permanently disable after one hard failure
        self._warned_once = False

    # ----- gallery side -----
    def build_gallery(self, gallery_rows_by_id: Dict[str, Row]) -> None:
        self.vecs_by_id = load_vectors_for_field("Gallery", self.name, expected_model_id=self.model_id)

    # ----- query side -----
    def prepare_query(self, q_row: Row) -> Any:
        if self._fatal_disabled:
            return None
        qid = (q_row.get("query_id", "") or "").strip()
        text = (q_row.get(self.name, "") or "")
        if not text.strip():
            return None
        try:
            return upsert_single_query_vector(qid, self.name, text, model_id=self.model_id)
        except Exception as e:
            if not self._warned_once:
                _log.warning("TextEmbeddingBGEScorer disabled for field=%s due to: %s", self.name, e)
                self._warned_once = True
            self._fatal_disabled = True
            return None

    def has_query_signal(self, q_state: Any) -> bool:
        return q_state is not None

    def score_pair(self, q_state: Any, gallery_id: str) -> Tuple[float, bool]:
        g = self.vecs_by_id.get(gallery_id)
        if g is None or q_state is None:
            return 0.0, False
        # cosine in [-1,1] -> [0,1]; vectors are normalized by encode(normalize_embeddings=True)
        s = float(np.dot(q_state, g))
        # numerica lguard
        s = 1.0 if s >1.0 else (-1.0 if s < -1.0 else s)
        return 0.5 * (s + 1.0), True
