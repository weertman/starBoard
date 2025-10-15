# src/search/embed_store.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple
import json, hashlib, os, importlib.util, logging
from datetime import datetime

import numpy as np
from sentence_transformers import SentenceTransformer

from src.data.archive_paths import gallery_root, queries_root

log = logging.getLogger("starBoard.search.embed_store")

Row = Dict[str, str]

# ---------- config ----------
DEFAULT_MODEL_ID = os.getenv("STARBOARD_EMBED_MODEL", "BAAI/bge-m3")
_MODEL_CACHE: Dict[str, SentenceTransformer] = {}

def _version_tuple(ver: str) -> tuple[int, int, int]:
    try:
        ver = ver.split("+", 1)[0]
        parts = (ver.split(".") + ["0", "0", "0"])[:3]
        return tuple(int(p) for p in parts)  # type: ignore[return-value]
    except Exception:
        return (0, 0, 0)

def embedding_backend_ready() -> tuple[bool, str]:
    """
    Returns (ok, reason). ok==True if either:
      - the 'safetensors' package is available (so we won't call torch.load), OR
      - torch >= 2.6 is installed (transformers considers it safe to use torch.load).
    """
    has_sft = importlib.util.find_spec("safetensors") is not None
    torch_ok = False
    try:
        import torch  # noqa
        torch_ok = _version_tuple(getattr(torch, "__version__", "0.0.0")) >= (2, 6, 0)
    except Exception:
        torch_ok = False

    if has_sft or torch_ok:
        return True, "ok"
    return False, "Install `safetensors` or upgrade `torch` to >=2.6 (transformers forbids torch.load on older torch)."

def _model(model_id: str) -> SentenceTransformer:
    ok, reason = embedding_backend_ready()
    if not ok:
        raise RuntimeError(f"Embedding backend unavailable: {reason}")
    m = _MODEL_CACHE.get(model_id)
    if m is None:
        # Device is chosen by sentence-transformers; it will prefer CUDA if available
        log.info("Loading SentenceTransformer: %s", model_id)
        m = SentenceTransformer(model_id)
        _MODEL_CACHE[model_id] = m
    return m

def _store_path(target: str) -> Path:
    t = (target or "").lower()
    root = gallery_root() if t == "gallery" else queries_root(prefer_new=True)
    root.mkdir(parents=True, exist_ok=True)
    return root / "metadata_embeddings.json"

def _sha1(s: str) -> str:
    return hashlib.sha1((s or "").encode("utf-8")).hexdigest()

def _load_store(path: Path) -> dict:
    if not path.exists():
        return {"model_id": "", "dims": 0, "fields": {}, "hashes": {}, "updated_utc": ""}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"model_id": "", "dims": 0, "fields": {}, "hashes": {}, "updated_utc": ""}

def _atomic_write_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    tmp.replace(path)

# ---------- public API ----------
def ensure_metadata_embeddings(
    target: str,
    rows_by_id: Dict[str, Row],
    *,
    fields: List[str],
    model_id: str = DEFAULT_MODEL_ID,
    force: bool = False,
) -> Tuple[int, int]:
    """
    Ensure metadata_embeddings.json has vectors for all (id, field) with non-empty text.
    Default (force=False): only create MISSING vectors; changed texts remain stale until forced.
    """
    ok, reason = embedding_backend_ready()
    if not ok:
        raise RuntimeError(reason)

    path = _store_path(target)
    store = _load_store(path)
    changed = False

    mdl = _model(model_id)
    dims = int(getattr(mdl, "get_sentence_embedding_dimension", lambda: 0)() or 0)

    reset_all = (store.get("model_id") != model_id) or (int(store.get("dims", 0)) != dims)

    fields_map: dict = store.setdefault("fields", {})
    hashes_map: dict = store.setdefault("hashes", {})

    n_done = 0
    n_cand = 0

    for field in fields:
        fvecs: dict = fields_map.setdefault(field, {})
        fhash: dict = hashes_map.setdefault(field, {})

        if force or reset_all:
            if fvecs or fhash:
                fvecs.clear(); fhash.clear(); changed = True

        for _id, row in rows_by_id.items():
            text = (row.get(field, "") or "")
            if not text.strip():
                if _id in fvecs or _id in fhash:
                    fvecs.pop(_id, None); fhash.pop(_id, None); changed = True
                continue

            n_cand += 1
            h = _sha1(text)

            if _id not in fvecs or _id not in fhash:
                vec = mdl.encode(text, normalize_embeddings=True)
                fvecs[_id] = [float(x) for x in np.asarray(vec, dtype=np.float32)]
                fhash[_id] = h
                changed = True
                n_done += 1
                continue

            prev_h = fhash.get(_id)
            if prev_h != h and force:
                vec = mdl.encode(text, normalize_embeddings=True)
                fvecs[_id] = [float(x) for x in np.asarray(vec, dtype=np.float32)]
                fhash[_id] = h
                changed = True
                n_done += 1

    if changed or reset_all:
        store["model_id"] = model_id
        store["dims"] = dims
        store["updated_utc"] = datetime.utcnow().isoformat() + "Z"
        _atomic_write_json(path, store)

    return n_done, n_cand

def load_vectors_for_field(target: str, field: str, expected_model_id: str = DEFAULT_MODEL_ID) -> Dict[str, np.ndarray]:
    path = _store_path(target)
    store = _load_store(path)
    if store.get("model_id") != expected_model_id:
        return {}
    arrs: Dict[str, np.ndarray] = {}
    for _id, vec in store.get("fields", {}).get(field, {}).items():
        arrs[_id] = np.asarray(vec, dtype=np.float32)
    return arrs

def upsert_single_query_vector(
    query_id: str, field: str, text: str, *,
    model_id: str = DEFAULT_MODEL_ID
) -> np.ndarray:
    ok, reason = embedding_backend_ready()
    if not ok:
        raise RuntimeError(reason)

    mdl = _model(model_id)
    vec = mdl.encode(text, normalize_embeddings=True)
    v = np.asarray(vec, dtype=np.float32)

    path = _store_path("Queries")
    store = _load_store(path)
    dims = int(getattr(mdl, "get_sentence_embedding_dimension", lambda: 0)() or 0)
    if store.get("model_id") != model_id or int(store.get("dims", 0)) != dims:
        store = {"model_id": model_id, "dims": dims, "fields": {}, "hashes": {}, "updated_utc": ""}

    store.setdefault("fields", {}).setdefault(field, {})[query_id] = [float(x) for x in v]
    store.setdefault("hashes", {}).setdefault(field, {})[query_id] = hashlib.sha1((text or "").encode("utf-8")).hexdigest()
    store["updated_utc"] = datetime.utcnow().isoformat() + "Z"
    _atomic_write_json(path, store)
    return v
