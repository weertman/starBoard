# src/search/embed_store.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Iterable
import json
import hashlib
import os
import importlib.util
import logging
from datetime import datetime

import numpy as np
from sentence_transformers import SentenceTransformer

from src.data.archive_paths import gallery_root, queries_root

log = logging.getLogger("starBoard.search.embed_store")

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------

# Default Sentence-Transformer model (fast and good for general retrieval).
# You can override with env var SB_EMBED_MODEL.
DEFAULT_MODEL_ID: str = os.environ.get("SB_EMBED_MODEL", "BAAI/bge-small-en-v1.5")

# Where to put the JSON stores (under each target's archive directory)
_STORE_FILENAME = "text_embeddings_bge.json"

# Cache for model instances
_MODEL_CACHE: Dict[str, SentenceTransformer] = {}

# Types
Row = Dict[str, str]


# --------------------------------------------------------------------------------------
# Backend readiness / model loading
# --------------------------------------------------------------------------------------
def embedding_backend_ready() -> tuple[bool, str]:
    """
    Check if the environment can safely load ST models.
    We accept either:
      - `safetensors` present, or
      - torch present (ideally >= 2.6 to avoid older torch.load restrictions in transformers)
    """
    try:
        has_safetensors = importlib.util.find_spec("safetensors") is not None
    except Exception:
        has_safetensors = False

    torch_ok = False
    try:
        if importlib.util.find_spec("torch") is not None:
            import torch
            # Avoid strict dependency on packaging; do a lightweight version check.
            def _parse(v: str) -> Tuple[int, int]:
                parts = (v.split("+", 1)[0]).split(".")
                major = int(parts[0]) if len(parts) > 0 and parts[0].isdigit() else 0
                minor = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
                return major, minor
            major, minor = _parse(getattr(torch, "__version__", "0.0"))
            torch_ok = (major, minor) >= (2, 6)
        else:
            torch_ok = False
    except Exception:
        # If we cannot import or parse, leave as False (safetensors can still save us)
        torch_ok = False

    if has_safetensors or torch_ok:
        return True, "ok"

    return False, "Install `safetensors` or upgrade `torch` to >= 2.6 (transformers forbids torch.load on older torch)."


def _model(model_id: str) -> SentenceTransformer:
    """Load or reuse a SentenceTransformer model instance."""
    ok, reason = embedding_backend_ready()
    if not ok:
        raise RuntimeError(f"Embedding backend unavailable: {reason}")

    m = _MODEL_CACHE.get(model_id)
    if m is None:
        log.info("Loading SentenceTransformer: %s", model_id)
        m = SentenceTransformer(model_id)
        _MODEL_CACHE[model_id] = m
    return m


# --------------------------------------------------------------------------------------
# Store I/O
# --------------------------------------------------------------------------------------
def _root_for(target: str) -> Path:
    t = target.lower()
    if t == "gallery":
        return gallery_root()
    if t == "queries":
        # Prefer the new folder if it exists; archive_paths.queries_root already does that.
        return queries_root(prefer_new=True)
    raise ValueError(f"Unknown target '{target}', expected 'Gallery' or 'Queries'.")


def _store_dir(target: str) -> Path:
    return _root_for(target) / "_embeddings"


def _store_path(target: str) -> Path:
    d = _store_dir(target)
    d.mkdir(parents=True, exist_ok=True)
    return d / _STORE_FILENAME


def _atomic_write_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, path)


def _load_store(path: Path) -> dict:
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            log.warning("Failed to load embedding store at %s: %s; starting fresh", str(path), e)
    return {}


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def _text_hash(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()


def _ensure_store_model(store: dict, model_id: str, dims: int) -> dict:
    if store.get("model_id") != model_id or int(store.get("dims", 0)) != int(dims):
        return {"model_id": model_id, "dims": int(dims), "fields": {}, "hashes": {}, "updated_utc": ""}
    # ensure expected keys exist
    store.setdefault("fields", {})
    store.setdefault("hashes", {})
    return store


def _as_float_list(vec: np.ndarray) -> List[float]:
    return [float(x) for x in np.asarray(vec, dtype=np.float32).reshape(-1)]


def _as_np_array(lst: Iterable[float]) -> np.ndarray:
    return np.asarray(list(lst), dtype=np.float32)


# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------
def ensure_metadata_embeddings(
    target: str,
    rows_by_id: Dict[str, Row],
    *,
    fields: List[str],
    force: bool = False,
    model_id: str = DEFAULT_MODEL_ID,
    batch_size: int = 64,
) -> Tuple[int, int]:
    """
    Ensure embeddings exist for all (id, field) that have nonempty text.
    Only (re)embeds if:
      - force=True, or
      - hash changed, or
      - vector missing / wrong dims, or
      - store model/dims changed.

    Returns
    -------
    (n_done, n_candidates)
      n_done:       number of vectors newly computed or refreshed this call
      n_candidates: number of (id, field) with non-empty text that were checked
    """
    ok, reason = embedding_backend_ready()
    if not ok:
        raise RuntimeError(f"Embedding backend unavailable: {reason}")

    mdl = _model(model_id)
    dims = int(getattr(mdl, "get_sentence_embedding_dimension", lambda: 0)() or 0)
    path = _store_path(target)
    store = _ensure_store_model(_load_store(path), model_id, dims)

    n_done = 0
    n_cand = 0
    any_changes = False

    # We process per field for memory locality & to enable batch encoding.
    for field in fields:
        field_map = store["fields"].setdefault(field, {})
        hash_map = store["hashes"].setdefault(field, {})

        # Gather candidates
        to_embed_ids: List[str] = []
        to_embed_texts: List[str] = []
        for _id, row in rows_by_id.items():
            text = (row.get(field, "") or "").strip()
            if not text:
                continue
            n_cand += 1

            want_hash = _text_hash(text)
            have_vec = field_map.get(_id)
            have_hash = hash_map.get(_id)

            needs = (
                force
                or (have_vec is None)
                or (not isinstance(have_vec, list))
                or (len(have_vec) != dims)
                or (have_hash != want_hash)
            )
            if needs:
                to_embed_ids.append(_id)
                to_embed_texts.append(text)

        if not to_embed_ids:
            continue

        # Batch encode
        log.info("Encoding %d %s/%s texts (batch=%d)", len(to_embed_ids), target, field, batch_size)
        vecs = mdl.encode(
            to_embed_texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        vecs = np.asarray(vecs, dtype=np.float32)

        # Upsert results
        for _id, v, text in zip(to_embed_ids, vecs, to_embed_texts):
            field_map[_id] = _as_float_list(v)
            hash_map[_id] = _text_hash(text)
            n_done += 1
        any_changes = True

    if any_changes:
        store["updated_utc"] = datetime.utcnow().isoformat() + "Z"
        _atomic_write_json(path, store)

    return n_done, n_cand


def load_vectors_for_field(
    target: str,
    field: str,
    expected_model_id: str = DEFAULT_MODEL_ID,
) -> Dict[str, np.ndarray]:
    """
    Load all vectors for a given target ('Gallery' or 'Queries') and field.
    Returns a mapping: id -> np.ndarray(float32)
    """
    path = _store_path(target)
    store = _load_store(path)
    if not store:
        return {}

    # If the stored model doesn't match expectations, treat as empty
    if store.get("model_id") != expected_model_id:
        raise RuntimeError(
            f"Embedding store at {path} was built with model_id={store.get('model_id')}, "
            f"but expected {expected_model_id}."
        )

    dims = int(store.get("dims", 0) or 0)
    if dims <= 0:
        return {}

    field_map = store.get("fields", {}).get(field, {}) or {}
    out: Dict[str, np.ndarray] = {}
    for _id, lst in field_map.items():
        try:
            if isinstance(lst, list) and len(lst) == dims:
                out[_id] = _as_np_array(lst)
        except Exception:
            # Skip corrupted entries
            continue
    return out


def upsert_single_query_vector(
    query_id: str,
    field: str,
    text: str,
    model_id: str = DEFAULT_MODEL_ID,
) -> np.ndarray:
    """
    Ensure a query embedding exists and is up-to-date for (query_id, field).
    Reuses the stored vector when the text hash and model/dims match;
    otherwise (re)embeds and persists.

    Returns
    -------
    np.ndarray (float32)  # L2-normalized vector as produced by SentenceTransformer.encode(..., normalize_embeddings=True)
    """
    ok, reason = embedding_backend_ready()
    if not ok:
        raise RuntimeError(f"Embedding backend unavailable: {reason}")

    mdl = _model(model_id)
    dims = int(getattr(mdl, "get_sentence_embedding_dimension", lambda: 0)() or 0)

    path = _store_path("Queries")
    store = _ensure_store_model(_load_store(path), model_id, dims)

    fields_map = store.setdefault("fields", {}).setdefault(field, {})
    hashes_map = store.setdefault("hashes", {}).setdefault(field, {})

    new_hash = _text_hash(text)

    # Fast path: reuse stored vector if text/model/dims match
    v_existing = fields_map.get(query_id)
    if hashes_map.get(query_id) == new_hash and isinstance(v_existing, list) and len(v_existing) == dims:
        return _as_np_array(v_existing)

    # Otherwise, (re)embed and upsert
    vec = mdl.encode(text, normalize_embeddings=True)
    v = np.asarray(vec, dtype=np.float32)

    fields_map[query_id] = _as_float_list(v)
    hashes_map[query_id] = new_hash
    store["updated_utc"] = datetime.utcnow().isoformat() + "Z"
    _atomic_write_json(path, store)

    return v
