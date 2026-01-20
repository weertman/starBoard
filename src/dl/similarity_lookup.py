"""
Fast similarity lookup from precomputed matrices.

Provides O(1) lookups for query-gallery similarity scores.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .registry import DLRegistry

log = logging.getLogger("starBoard.dl.similarity_lookup")


class SimilarityLookup:
    """
    Fast lookup for precomputed similarity scores.
    
    Loads the precomputed similarity matrix and provides
    efficient query-gallery score lookups.
    """
    
    def __init__(self, model_key: str):
        self.model_key = model_key
        self._similarity_matrix: Optional[np.ndarray] = None
        self._query_ids: List[str] = []
        self._gallery_ids: List[str] = []
        self._query_to_idx: Dict[str, int] = {}
        self._gallery_to_idx: Dict[str, int] = {}
        self._loaded = False
    
    def load(self) -> bool:
        """Load the precomputed similarity data."""
        model_dir = DLRegistry.get_model_data_dir(self.model_key)
        similarity_dir = model_dir / "similarity"
        
        matrix_path = similarity_dir / "query_gallery_scores.npz"
        mapping_path = similarity_dir / "id_mapping.json"
        
        if not matrix_path.exists() or not mapping_path.exists():
            log.warning("Precomputed data not found for model %s", self.model_key)
            return False
        
        try:
            # Load similarity matrix
            data = np.load(matrix_path)
            self._similarity_matrix = data['similarity']
            
            # Load ID mapping
            with open(mapping_path, 'r', encoding='utf-8') as f:
                mapping = json.load(f)
            
            self._query_ids = mapping['query_ids']
            self._gallery_ids = mapping['gallery_ids']
            self._query_to_idx = {qid: i for i, qid in enumerate(self._query_ids)}
            self._gallery_to_idx = {gid: i for i, gid in enumerate(self._gallery_ids)}
            
            self._loaded = True
            log.info("Loaded similarity data: %d queries x %d gallery", 
                     len(self._query_ids), len(self._gallery_ids))
            return True
            
        except Exception as e:
            log.error("Failed to load similarity data: %s", e)
            return False
    
    def is_loaded(self) -> bool:
        """Check if similarity data is loaded."""
        return self._loaded
    
    def get_scores_for_query(self, query_id: str) -> Dict[str, float]:
        """
        Get similarity scores for all gallery items for a given query.
        
        Args:
            query_id: The query ID to look up
            
        Returns:
            Dictionary mapping gallery_id to similarity score
        """
        if not self._loaded:
            return {}
        
        if query_id not in self._query_to_idx:
            log.warning("Query ID not in precomputed data: %s", query_id)
            return {}
        
        q_idx = self._query_to_idx[query_id]
        scores = self._similarity_matrix[q_idx]
        
        return {gid: float(scores[i]) for i, gid in enumerate(self._gallery_ids)}
    
    def get_score(self, query_id: str, gallery_id: str) -> Optional[float]:
        """
        Get the similarity score for a specific query-gallery pair.
        
        Args:
            query_id: The query ID
            gallery_id: The gallery ID
            
        Returns:
            Similarity score, or None if not found
        """
        if not self._loaded:
            return None
        
        if query_id not in self._query_to_idx:
            return None
        if gallery_id not in self._gallery_to_idx:
            return None
        
        q_idx = self._query_to_idx[query_id]
        g_idx = self._gallery_to_idx[gallery_id]
        
        return float(self._similarity_matrix[q_idx, g_idx])
    
    def get_ranked_gallery(self, query_id: str, top_k: Optional[int] = None) -> List[Tuple[str, float]]:
        """
        Get gallery items ranked by similarity to a query.
        
        Args:
            query_id: The query ID
            top_k: Optional limit on number of results
            
        Returns:
            List of (gallery_id, score) tuples, sorted by descending score
        """
        scores = self.get_scores_for_query(query_id)
        if not scores:
            return []
        
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        
        if top_k is not None:
            ranked = ranked[:top_k]
        
        return ranked
    
    def has_query(self, query_id: str) -> bool:
        """Check if a query ID is in the precomputed data."""
        return query_id in self._query_to_idx
    
    def has_gallery(self, gallery_id: str) -> bool:
        """Check if a gallery ID is in the precomputed data."""
        return gallery_id in self._gallery_to_idx
    
    def get_query_ids(self) -> List[str]:
        """Get all query IDs in the precomputed data."""
        return list(self._query_ids)
    
    def get_gallery_ids(self) -> List[str]:
        """Get all gallery IDs in the precomputed data."""
        return list(self._gallery_ids)


class ImageSimilarityLookup:
    """
    Fast lookup for image-level similarity scores.
    
    Provides efficient lookups for:
    - Best matching gallery image for a specific query image
    - Ranking gallery identities by best-match to a query image
    
    Note: Stored paths are cache paths (precompute_cache/...), but lookups
    can use either cache paths OR original archive paths (auto-converted).
    """
    
    def __init__(self, model_key: str):
        self.model_key = model_key
        self._image_similarity: Optional[np.ndarray] = None
        self._query_image_index: List[dict] = []  # [{id, local_idx, path}, ...]
        self._gallery_image_index: List[dict] = []
        self._query_path_to_idx: Dict[str, int] = {}
        self._gallery_path_to_idx: Dict[str, int] = {}
        # Additional lookup by (id, filename_stem) for original path matching
        self._query_id_stem_to_idx: Dict[Tuple[str, str], int] = {}
        self._gallery_id_stem_to_idx: Dict[Tuple[str, str], int] = {}
        # Precomputed identity grouping for fast vectorized max-finding
        # Gallery images are contiguous by identity (sorted during precomputation)
        self._identity_slices: Dict[str, Tuple[int, int]] = {}  # {gid: (start, end)}
        self._identity_order: List[str] = []  # Ordered list of gallery IDs
        self._loaded = False
    
    def load(self) -> bool:
        """Load the precomputed image-level similarity data."""
        model_dir = DLRegistry.get_model_data_dir(self.model_key)
        similarity_dir = model_dir / "similarity"
        
        matrix_path = similarity_dir / "image_similarity_matrix.npz"
        query_index_path = similarity_dir / "query_image_index.json"
        gallery_index_path = similarity_dir / "gallery_image_index.json"
        
        if not matrix_path.exists():
            log.warning("Image similarity matrix not found for model %s", self.model_key)
            return False
        
        try:
            # Load image similarity matrix
            data = np.load(matrix_path)
            self._image_similarity = data['similarity']
            
            # Load image indices
            with open(query_index_path, 'r', encoding='utf-8') as f:
                self._query_image_index = json.load(f)
            with open(gallery_index_path, 'r', encoding='utf-8') as f:
                self._gallery_image_index = json.load(f)
            
            # Build path-to-index lookups
            self._query_path_to_idx = {
                item['path']: i for i, item in enumerate(self._query_image_index)
            }
            self._gallery_path_to_idx = {
                item['path']: i for i, item in enumerate(self._gallery_image_index)
            }
            
            # Build (id, filename_stem) lookups for original path matching
            # Cache paths use .png extension, originals may use .jpg/.jpeg etc
            for i, item in enumerate(self._query_image_index):
                stem = Path(item['path']).stem  # filename without extension
                self._query_id_stem_to_idx[(item['id'], stem)] = i
            
            for i, item in enumerate(self._gallery_image_index):
                stem = Path(item['path']).stem
                self._gallery_id_stem_to_idx[(item['id'], stem)] = i
            
            # Build identity slices for fast vectorized max-finding
            # Gallery images are contiguous by identity (sorted during precomputation)
            self._identity_slices = {}
            self._identity_order = []
            current_id = None
            start_idx = 0
            for i, item in enumerate(self._gallery_image_index):
                gid = item['id']
                if gid != current_id:
                    if current_id is not None:
                        self._identity_slices[current_id] = (start_idx, i)
                        self._identity_order.append(current_id)
                    current_id = gid
                    start_idx = i
            # Don't forget the last identity
            if current_id is not None:
                self._identity_slices[current_id] = (start_idx, len(self._gallery_image_index))
                self._identity_order.append(current_id)
            
            self._loaded = True
            log.info("Loaded image similarity: %d query images x %d gallery images (%d identities)",
                     len(self._query_image_index), len(self._gallery_image_index),
                     len(self._identity_order))
            return True
            
        except Exception as e:
            log.error("Failed to load image similarity data: %s", e)
            return False
    
    def is_loaded(self) -> bool:
        """Check if image similarity data is loaded."""
        return self._loaded
    
    def _resolve_query_image_idx(self, image_path: str) -> Optional[int]:
        """
        Resolve a query image path to its index.
        
        Handles both cache paths and original archive paths.
        Original paths are matched by (identity_id, filename_stem).
        """
        # Try direct cache path lookup first
        if image_path in self._query_path_to_idx:
            return self._query_path_to_idx[image_path]
        
        # Try to extract identity and stem from original path
        # Original paths look like: .../Queries/ID/encounter/photo.jpg
        # or: .../Gallery/ID/encounter/photo.jpg
        try:
            path = Path(image_path)
            stem = path.stem  # filename without extension
            
            # Walk up to find the identity ID (parent of encounter dir)
            # Path structure: .../target/ID/encounter/file.ext
            parts = path.parts
            
            # Find "Queries" or "queries" in path
            for i, part in enumerate(parts):
                if part.lower() in ('queries', 'gallery'):
                    if i + 1 < len(parts):
                        identity_id = parts[i + 1]
                        key = (identity_id, stem)
                        if key in self._query_id_stem_to_idx:
                            return self._query_id_stem_to_idx[key]
                    break
            
            # Fallback: try just the stem with all identities
            for (id_str, s), idx in self._query_id_stem_to_idx.items():
                if s == stem:
                    return idx
                    
        except Exception:
            pass
        
        return None
    
    def get_scores_for_query_image(self, query_image_path: str) -> Dict[str, float]:
        """
        Get similarity scores from a query image to all gallery images.
        
        Args:
            query_image_path: Path to the query image (cache or original)
            
        Returns:
            Dict mapping gallery image path to similarity score
        """
        if not self._loaded:
            return {}
        
        q_idx = self._resolve_query_image_idx(query_image_path)
        if q_idx is None:
            log.warning("Query image not in precomputed data: %s", query_image_path)
            return {}
        
        scores = self._image_similarity[q_idx]
        
        return {
            self._gallery_image_index[i]['path']: float(scores[i])
            for i in range(len(self._gallery_image_index))
        }
    
    def rank_gallery_by_query_image(
        self, 
        query_image_path: str,
        top_k: Optional[int] = None
    ) -> List[Tuple[str, float, str, int]]:
        """
        Rank gallery IDENTITIES by best-match to a specific query image.
        
        For each gallery identity, finds the maximum similarity score
        from the query image to any image in that gallery identity.
        
        Uses precomputed identity slices for vectorized max-finding,
        which is significantly faster than iterating through all images.
        
        Args:
            query_image_path: Path to the query image (cache or original)
            top_k: Optional limit on results
            
        Returns:
            List of (gallery_id, best_score, best_gallery_image_path, local_idx) tuples,
            sorted by descending score. local_idx is the index of the best image
            within that gallery identity's image list (for direct indexing).
        """
        if not self._loaded:
            return []
        
        q_idx = self._resolve_query_image_idx(query_image_path)
        if q_idx is None:
            log.debug("Query image not found for ranking: %s", query_image_path)
            return []
        
        scores = self._image_similarity[q_idx]
        
        # Use precomputed identity slices for vectorized max-finding
        # Each identity's images are contiguous in the array (sorted during precomputation)
        results: List[Tuple[str, float, str, int]] = []
        
        for gid in self._identity_order:
            start, end = self._identity_slices[gid]
            slice_scores = scores[start:end]
            
            # NumPy argmax is vectorized and much faster than Python loop
            local_argmax = int(np.argmax(slice_scores))
            best_score = float(slice_scores[local_argmax])
            global_idx = start + local_argmax
            
            item = self._gallery_image_index[global_idx]
            results.append((gid, best_score, item['path'], item.get('local_idx', 0)))
        
        # Sort by score descending
        results.sort(key=lambda x: -x[1])
        
        if top_k is not None:
            results = results[:top_k]
        
        return results
    
    def get_best_match_for_query_image(
        self,
        query_image_path: str
    ) -> Optional[Tuple[str, float, str]]:
        """
        Get the best matching gallery image for a query image.
        
        Args:
            query_image_path: Path to the query image (cache or original)
            
        Returns:
            (gallery_id, score, gallery_image_path) or None
        """
        if not self._loaded:
            return None
        
        q_idx = self._resolve_query_image_idx(query_image_path)
        if q_idx is None:
            return None
        
        scores = self._image_similarity[q_idx]
        
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        best_item = self._gallery_image_index[best_idx]
        
        return (best_item['id'], best_score, best_item['path'])
    
    def has_query_image(self, image_path: str) -> bool:
        """Check if a query image is in the precomputed data."""
        return self._resolve_query_image_idx(image_path) is not None
    
    def has_gallery_image(self, image_path: str) -> bool:
        """Check if a gallery image is in the precomputed data."""
        return image_path in self._gallery_path_to_idx
    
    def get_query_images_for_id(self, query_id: str) -> List[str]:
        """Get all query image paths for an identity."""
        return [
            item['path'] for item in self._query_image_index
            if item['id'] == query_id
        ]
    
    def get_gallery_images_for_id(self, gallery_id: str) -> List[str]:
        """Get all gallery image paths for an identity."""
        return [
            item['path'] for item in self._gallery_image_index
            if item['id'] == gallery_id
        ]


# Cache of loaded lookups
_lookup_cache: Dict[str, SimilarityLookup] = {}
_image_lookup_cache: Dict[str, ImageSimilarityLookup] = {}


def get_lookup(model_key: str) -> Optional[SimilarityLookup]:
    """
    Get a SimilarityLookup for the given model.
    
    Results are cached.
    """
    if model_key in _lookup_cache:
        lookup = _lookup_cache[model_key]
        if lookup.is_loaded():
            return lookup
    
    lookup = SimilarityLookup(model_key)
    if lookup.load():
        _lookup_cache[model_key] = lookup
        return lookup
    
    return None


def clear_cache():
    """Clear the lookup cache."""
    _lookup_cache.clear()
    _image_lookup_cache.clear()


def get_active_lookup() -> Optional[SimilarityLookup]:
    """Get a lookup for the currently active model."""
    registry = DLRegistry.load()
    if not registry.active_model:
        return None
    return get_lookup(registry.active_model)


def get_image_lookup(model_key: str) -> Optional[ImageSimilarityLookup]:
    """
    Get an ImageSimilarityLookup for the given model.
    
    Results are cached.
    """
    if model_key in _image_lookup_cache:
        lookup = _image_lookup_cache[model_key]
        if lookup.is_loaded():
            return lookup
    
    lookup = ImageSimilarityLookup(model_key)
    if lookup.load():
        _image_lookup_cache[model_key] = lookup
        return lookup
    
    return None


def get_active_image_lookup() -> Optional[ImageSimilarityLookup]:
    """Get an image lookup for the currently active model."""
    registry = DLRegistry.load()
    if not registry.active_model:
        return None
    return get_image_lookup(registry.active_model)

