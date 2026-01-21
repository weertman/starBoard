# src/search/engine.py
from __future__ import annotations
from typing import Dict, List, Tuple, Any, Set
from dataclasses import dataclass
import logging
import os
import time
import math
from collections import Counter

from src.data.archive_paths import (
    metadata_csv_paths_for_read,
    id_column_name,
    QUERIES_HEADER,
    GALLERY_HEADER,
    archive_root,
    metadata_csv_for,
)
from src.data.csv_io import read_rows_multi, last_row_per_id, normalize_id_value, ensure_header
from .interfaces import FieldScorer
from .fields_numeric import NumericGaussianScorer
from .fields_categorical import CategoricalMatchScorer
from .fields_set_jaccard import SetJaccardScorer
from .fields_short_arm_code import ShortArmCodeScorer
from .fields_text_ngrams import TextNgramScorer
from .fields_color import ColorSpaceScorer
from .fields_text_embed_bge import TextEmbeddingBGEScorer
from .embed_store import ensure_metadata_embeddings, embedding_backend_ready, DEFAULT_MODEL_ID

logger = logging.getLogger("starBoard.search.engine")

Row = Dict[str, str]

# =============================================================================
# V2 SCHEMA FIELD DEFINITIONS
# =============================================================================

# Numeric fields - continuous measurements
NUMERIC_FIELDS = [
    "num_apparent_arms",
    "num_total_arms",
    "tip_to_tip_size_cm",
]

# Ordinal categorical fields - stored as numeric values (0, 1, 2, 3...)
# These use numeric Gaussian scoring since order matters
ORDINAL_FIELDS = [
    "stripe_order",        # 0=None, 1=Mixed, 2=Irregular, 3=Regular
    "stripe_prominence",   # 0=None, 1=Weak, 2=Medium, 3=Strong, 4=Strongest
    "stripe_extent",       # 0.0=None, 0.25=Quarter, 0.5=Halfway, 0.75=Three quarters, 1.0=Full
    "stripe_thickness",    # 0=None, 1=Thin, 2=Medium, 3=Thick
    "arm_thickness",       # 0=Thin, 1=Medium, 2=Thick
    "rosette_prominence",  # 0=Weak, 1=Medium, 2=Strong
    "reticulation_order",  # 0=None, 1=Mixed, 2=Meandering, 3=Train tracks
]

# Color categorical fields - exact string match
COLOR_FIELDS = [
    "stripe_color",
    "arm_color",
    "central_disc_color",
    "papillae_central_disc_color",
    "rosette_color",
    "papillae_stripe_color",
    "madreporite_color",
    "overall_color",
]

# Set/code fields - Jaccard similarity on tokenized sets
SET_FIELDS = [
    "short_arm_code",
]

# Free text fields - embedding or n-gram similarity
TEXT_FIELDS = [
    "location",
    "unusual_observation",
    "health_observation",
]

# All fields combined for iteration
ALL_FIELDS: List[str] = (
    NUMERIC_FIELDS + ORDINAL_FIELDS + COLOR_FIELDS + SET_FIELDS + TEXT_FIELDS
)


@dataclass
class RankItem:
    gallery_id: str
    score: float
    field_breakdown: Dict[str, float]
    k_contrib: int


class FirstOrderSearchEngine:
    def __init__(self):
        self._gallery_rows_by_id: Dict[str, Row] = {}
        self._queries_rows_by_id: Dict[str, Row] = {}
        self._built = False
        self._text_backend: str = "auto"  # "bge" | "ngrams"
        self.scorers: Dict[str, FieldScorer] = {}

    # ---------------- data loading ----------------
    def _load_latest_rows(self, target: str) -> Dict[str, Row]:
        id_col = id_column_name(target)
        rows = read_rows_multi(metadata_csv_paths_for_read(target))
        latest = last_row_per_id(rows, id_col)
        out: Dict[str, Row] = {}
        for _id, r in latest.items():
            r = dict(r)
            r[id_col] = _id
            out[_id] = r
        return out

    def _build_scorers(self, use_bge: bool) -> None:
        self.scorers = {}
        
        # Numeric fields - Gaussian decay based on MAD
        for f in NUMERIC_FIELDS:
            self.scorers[f] = NumericGaussianScorer(f)
        
        # Ordinal fields - also use Gaussian scoring since order matters
        for f in ORDINAL_FIELDS:
            self.scorers[f] = NumericGaussianScorer(f)
        
        # Color fields - perceptual similarity in LAB color space
        for f in COLOR_FIELDS:
            self.scorers[f] = ColorSpaceScorer(f, threshold=50.0, fallback_to_exact=True)
        
        # Set fields - specialized scorers
        for f in SET_FIELDS:
            if f == "short_arm_code":
                # Fuzzy position-aware matching for short arm codes
                self.scorers[f] = ShortArmCodeScorer(f, sigma=1.0, default_n_arms=5)
            else:
                # Generic Jaccard for other set fields
                self.scorers[f] = SetJaccardScorer(f)
        
        # Text fields - BGE embeddings or n-gram fallback
        if use_bge:
            for f in TEXT_FIELDS:
                self.scorers[f] = TextEmbeddingBGEScorer(f, model_id=DEFAULT_MODEL_ID)
            self._text_backend = "bge"
        else:
            for f in TEXT_FIELDS:
                self.scorers[f] = TextNgramScorer(f)
            self._text_backend = "ngrams"
        
        logger.info("text_backend=%s", self._text_backend)

    def rebuild(self, *, recompute_embeddings: bool = False) -> None:
        # Guard: skip redundant rebuild if already built and not forcing embeddings
        if self._built and not recompute_embeddings:
            logger.debug("rebuild skipped (already built, recompute_embeddings=False)")
            return
        
        # Proactively ensure/upgrade canonical metadata CSVs so new fields exist
        try:
            for t in ("Gallery", "Queries"):
                path, hdr = metadata_csv_for(t)
                ensure_header(path, hdr)
        except Exception as e:
            logger.warning("ensure_header during rebuild failed: %s", e)

        self._gallery_rows_by_id = self._load_latest_rows("Gallery")
        self._queries_rows_by_id = self._load_latest_rows("Queries")
        logger.info(
            "engine_rebuild start gallery_ids=%d query_ids=%d",
            len(self._gallery_rows_by_id), len(self._queries_rows_by_id)
        )

        ok, reason = embedding_backend_ready()
        self._build_scorers(use_bge=ok)

        # Only touch embeddings if backend is healthy
        if ok:
            try:
                g_done, g_cand = ensure_metadata_embeddings(
                    "Gallery", self._gallery_rows_by_id, fields=TEXT_FIELDS, force=recompute_embeddings
                )
                q_done, q_cand = ensure_metadata_embeddings(
                    "Queries", self._queries_rows_by_id, fields=TEXT_FIELDS, force=recompute_embeddings
                )
                logger.info(
                    "ensure_embeddings gallery=%d/%d queries=%d/%d force=%s",
                    g_done, g_cand, q_done, q_cand, recompute_embeddings
                )
            except Exception as e:
                # Fall back to n‑grams if embedding ensure failed
                logger.warning("ensure_embeddings failed: %s — falling back to n‑grams", e)
                self._build_scorers(use_bge=False)

        # Build per-field gallery indices
        for name, scorer in self.scorers.items():
            scorer.build_gallery(self._gallery_rows_by_id)

        # Coverage snapshots
        g_total = len(self._gallery_rows_by_id) or 1

        def _nonempty_count(field: str) -> int:
            return sum(1 for r in self._gallery_rows_by_id.values() if (r.get(field, "") or "").strip())

        for f in ALL_FIELDS:
            c = _nonempty_count(f)
            logger.info("engine_rebuild coverage field=%s gallery_nonempty=%d/%d", f, c, g_total)

        # Numeric and ordinal stats
        for f in NUMERIC_FIELDS + ORDINAL_FIELDS:
            sc = self.scorers[f]  # type: ignore
            n = len(getattr(sc, "values_by_id", {}))
            med = getattr(sc, "med", 0.0)
            mad = getattr(sc, "mad", 0.0)
            logger.info("engine_rebuild numeric field=%s n=%d med=%.3f mad=%.3f", f, n, med, mad)

        self._built = True

    def rebuild_if_needed(self) -> None:
        if not self._built:
            self.rebuild()
    
    def reset_built(self) -> None:
        """Reset the _built flag to force a full rebuild on next call."""
        self._built = False
        logger.debug("engine._built reset to False")

    # ---------------- ranking ----------------
    def _query_row(self, query_id: str) -> Row | None:
        return self._queries_rows_by_id.get(normalize_id_value(query_id))

    def rank(
        self,
        query_id: str,
        *,
        include_fields: Set[str] | None = None,
        equalize_weights: bool = True,
        weights: Dict[str, float] | None = None,
        top_k: int = 50,
        numeric_offsets: Dict[str, float] | None = None,
    ) -> List[RankItem]:
        self.rebuild_if_needed()
        q_row = self._query_row(query_id)
        if not q_row:
            logger.warning("rank_start qid=%s result=NO_QUERY", query_id)
            return []

        if include_fields is None or not include_fields:
            include_fields = set(ALL_FIELDS)

        # ----- Apply numeric offsets to a COPY of the query row -----
        # Offsets can apply to both numeric and ordinal fields
        numeric_like_fields = set(NUMERIC_FIELDS) | set(ORDINAL_FIELDS)
        q_row_eff = dict(q_row)
        if numeric_offsets:
            for f, off in numeric_offsets.items():
                try:
                    if f in numeric_like_fields:
                        v_raw = (q_row_eff.get(f, "") or "").strip()
                        if not v_raw:
                            continue
                        qv = float(v_raw)
                        ov = float(off)
                        if ov != 0.0 and not math.isnan(qv):
                            q_row_eff[f] = f"{(qv + ov):.12g}"
                except Exception:
                    # On any parsing error, leave as‑is
                    pass

        q_states: Dict[str, Any] = {}
        active_fields: List[str] = []
        for f in include_fields:
            scorer = self.scorers.get(f)
            if not scorer:
                continue
            # IMPORTANT: prepare against the offset‑adjusted query
            qs = scorer.prepare_query(q_row_eff)
            if scorer.has_query_signal(qs):
                q_states[f] = qs
                active_fields.append(f)

        if not active_fields:
            logger.warning("rank_start qid=%s include=%s active=0 -> empty", query_id, sorted(include_fields))
            return []

        eff_w: Dict[str, float] = {}
        if equalize_weights:
            eff_w = {f: 1.0 for f in active_fields}
        else:
            weights = weights or {}
            for f in active_fields:
                w = float(weights.get(f, 1.0))
                eff_w[f] = max(0.0, w)

        off_dbg = {k: float(v) for k, v in (numeric_offsets or {}).items() if abs(float(v)) > 0}
        logger.info(
            "rank_start qid=%s top_k=%d include=%s active=%s weights=%s offsets=%s",
            query_id, top_k, sorted(include_fields), sorted(active_fields),
            {k: round(v, 3) for k, v in eff_w.items()}, off_dbg
        )

        items: List[RankItem] = []
        for gid in self._gallery_rows_by_id.keys():
            num = 0.0
            den = 0.0
            breakdown: Dict[str, float] = {}
            k = 0
            for f in active_fields:
                sc = self.scorers[f]
                s, present = sc.score_pair(q_states[f], gid)
                if present:
                    w = eff_w[f]
                    num += w * s
                    den += w
                    breakdown[f] = float(s)
                    k += 1
            score = (num / den) if den > 0 else 0.0
            items.append(RankItem(gallery_id=gid, score=float(score), field_breakdown=breakdown, k_contrib=k))

        items.sort(key=lambda it: it.score, reverse=True)

        hist = Counter(it.k_contrib for it in items)
        logger.info("rank_k_contrib_hist qid=%s %s", query_id, dict(sorted(hist.items())))

        preview_n = min(10, len(items))
        for i in range(preview_n):
            it = items[i]
            parts = " ".join(f"{k}={it.field_breakdown[k]:.3f}" for k in sorted(it.field_breakdown))
            logger.info("rank_top qid=%s i=%d gid=%s score=%.3f k=%d %s",
                        query_id, i + 1, it.gallery_id, it.score, it.k_contrib, parts)

        if os.getenv("STARBOARD_DUMP_RANK_CSV"):
            try:
                logs_dir = archive_root() / "logs"
                logs_dir.mkdir(parents=True, exist_ok=True)
                out_path = logs_dir / f"first_order_{normalize_id_value(query_id)}_{int(time.time())}.csv"
                import csv
                with out_path.open("w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(["gallery_id", "score", "k_contrib", "fields"])
                    for it in items[:max(0, top_k)]:
                        fld = " ".join(f"{k}:{it.field_breakdown[k]:.4f}" for k in sorted(it.field_breakdown))
                        w.writerow([it.gallery_id, f"{it.score:.6f}", it.k_contrib, fld])
                logger.info("rank_dump_csv qid=%s path=%s rows=%d", query_id, str(out_path), min(top_k, len(items)))
            except Exception as e:
                logger.warning("rank_dump_csv_failed qid=%s err=%s", query_id, e)

        return items[: max(0, top_k)]
