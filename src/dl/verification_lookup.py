"""
Fast verification score lookup from precomputed matrices.

Provides O(1) lookups for query-gallery verification scores.
Verification scores represent P(same individual) from the cross-attention model.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .registry import DLRegistry

log = logging.getLogger("starBoard.dl.verification_lookup")


class VerificationLookup:
    """
    Fast lookup for precomputed verification scores.
    
    Loads the precomputed verification matrix (best-photo comparisons)
    and provides efficient query-gallery score lookups.
    
    Unlike SimilarityLookup which uses embedding cosine distances,
    this provides P(same individual) from the verification model.
    """
    
    def __init__(self, verification_model_key: str):
        self.model_key = verification_model_key
        self._verification_matrix: Optional[np.ndarray] = None
        self._query_ids: List[str] = []
        self._gallery_ids: List[str] = []
        self._query_to_idx: Dict[str, int] = {}
        self._gallery_to_idx: Dict[str, int] = {}
        self._best_photos: Dict[str, Dict[str, str]] = {}  # {target: {id: path}}
        self._metadata: Dict = {}
        self._loaded = False
    
    def load(self) -> bool:
        """Load the precomputed verification data."""
        model_dir = DLRegistry.get_verification_model_data_dir(self.model_key)
        verification_dir = model_dir / "verification"
        
        matrix_path = verification_dir / "verification_scores.npz"
        mapping_path = verification_dir / "id_mapping.json"
        best_photos_path = verification_dir / "best_photos.json"
        metadata_path = verification_dir / "metadata.json"
        
        if not matrix_path.exists() or not mapping_path.exists():
            log.warning("Verification data not found for model %s", self.model_key)
            return False
        
        try:
            # Load verification matrix
            data = np.load(matrix_path)
            self._verification_matrix = data['verification']
            
            # Load ID mapping
            with open(mapping_path, 'r', encoding='utf-8') as f:
                mapping = json.load(f)
            
            self._query_ids = mapping['query_ids']
            self._gallery_ids = mapping['gallery_ids']
            self._query_to_idx = {qid: i for i, qid in enumerate(self._query_ids)}
            self._gallery_to_idx = {gid: i for i, gid in enumerate(self._gallery_ids)}
            
            # Load best photos mapping (optional)
            if best_photos_path.exists():
                with open(best_photos_path, 'r', encoding='utf-8') as f:
                    self._best_photos = json.load(f)
            
            # Load metadata (optional)
            if metadata_path.exists():
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    self._metadata = json.load(f)
            
            self._loaded = True
            log.info(
                "Loaded verification data: %d queries x %d gallery",
                len(self._query_ids), len(self._gallery_ids)
            )
            return True
            
        except Exception as e:
            log.error("Failed to load verification data: %s", e)
            return False
    
    def is_loaded(self) -> bool:
        """Check if verification data is loaded."""
        return self._loaded
    
    def get_scores_for_query(self, query_id: str) -> Dict[str, float]:
        """
        Get verification scores for all gallery items for a given query.
        
        Args:
            query_id: The query ID to look up
            
        Returns:
            Dictionary mapping gallery_id to verification score P(same)
        """
        if not self._loaded:
            return {}
        
        if query_id not in self._query_to_idx:
            log.warning("Query ID not in verification data: %s", query_id)
            return {}
        
        q_idx = self._query_to_idx[query_id]
        scores = self._verification_matrix[q_idx]
        
        return {gid: float(scores[i]) for i, gid in enumerate(self._gallery_ids)}
    
    def get_score(self, query_id: str, gallery_id: str) -> Optional[float]:
        """
        Get the verification score for a specific query-gallery pair.
        
        Args:
            query_id: The query ID
            gallery_id: The gallery ID
            
        Returns:
            Verification score P(same), or None if not found
        """
        if not self._loaded:
            return None
        
        if query_id not in self._query_to_idx:
            return None
        if gallery_id not in self._gallery_to_idx:
            return None
        
        q_idx = self._query_to_idx[query_id]
        g_idx = self._gallery_to_idx[gallery_id]
        
        return float(self._verification_matrix[q_idx, g_idx])
    
    def get_ranked_gallery(
        self,
        query_id: str,
        top_k: Optional[int] = None
    ) -> List[Tuple[str, float]]:
        """
        Get gallery items ranked by verification score for a query.
        
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
    
    def get_best_photo_path(self, target: str, identity_id: str) -> Optional[str]:
        """
        Get the best photo path used for an identity.
        
        Args:
            target: "Gallery" or "Queries"
            identity_id: The identity ID
            
        Returns:
            Path to the best photo used for verification, or None
        """
        target_key = target.lower()
        if target_key not in self._best_photos:
            return None
        return self._best_photos[target_key].get(identity_id)
    
    def get_metadata(self) -> Dict:
        """Get metadata about the verification computation."""
        return self._metadata.copy()


# Cache of loaded lookups
_verification_cache: Dict[str, VerificationLookup] = {}


def get_verification_lookup(model_key: str) -> Optional[VerificationLookup]:
    """
    Get a VerificationLookup for the given model.
    
    Results are cached.
    """
    if model_key in _verification_cache:
        lookup = _verification_cache[model_key]
        if lookup.is_loaded():
            return lookup
    
    lookup = VerificationLookup(model_key)
    if lookup.load():
        _verification_cache[model_key] = lookup
        return lookup
    
    return None


def get_active_verification_lookup() -> Optional[VerificationLookup]:
    """Get a lookup for the currently active verification model."""
    registry = DLRegistry.load()
    if not registry.active_verification_model:
        return None
    return get_verification_lookup(registry.active_verification_model)


def clear_verification_cache():
    """Clear the verification lookup cache."""
    _verification_cache.clear()




