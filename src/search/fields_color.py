# src/search/fields_color.py
"""
Perceptual color similarity scorer for color categorical fields.

Instead of binary exact-match (1.0 or 0.0), this scorer uses:
- LAB color space for perceptual uniformity
- Delta E (CIE2000) for measuring color difference
- Exponential decay to map distance to similarity score

This means "orange" vs "red-orange" will have high similarity,
while "orange" vs "blue" will have low similarity.
"""
from __future__ import annotations
from typing import Dict, Any, Tuple, Optional
import logging

from .interfaces import Row
from src.data.color_config import get_color_config, delta_e_cie2000, rgb_to_lab

_log = logging.getLogger("starBoard.search.field.color")


class ColorSpaceScorer:
    """
    Perceptual color similarity scorer using LAB color space.
    
    Implements the FieldScorer protocol from src/search/interfaces.py.
    
    Similarity is computed as:
        sim = exp(-deltaE / threshold)
    
    Where deltaE is the CIE2000 perceptual color difference.
    
    With threshold=50 (default):
        deltaE=0  -> 1.00 (identical)
        deltaE=5  -> 0.90 (very similar, e.g., orange vs dark-orange)
        deltaE=10 -> 0.82 (similar, e.g., red vs orange-red)
        deltaE=25 -> 0.61 (somewhat similar)
        deltaE=50 -> 0.37 (different)
        deltaE=100 -> 0.14 (very different, e.g., red vs blue)
    """
    
    def __init__(
        self,
        field: str,
        threshold: float = 50.0,
        fallback_to_exact: bool = True,
    ):
        """
        Initialize the color scorer.
        
        Args:
            field: Field name (e.g., "stripe_color", "arm_color")
            threshold: Delta E threshold for similarity decay. Higher = more lenient.
            fallback_to_exact: If True, use exact match (0/1) when color isn't in database.
                              If False, return 0.0 for unknown colors.
        """
        self.name = field
        self.threshold = threshold
        self.fallback_to_exact = fallback_to_exact
        
        self._color_config = None  # Lazy load
        self._gallery_colors: Dict[str, str] = {}  # gid -> color name (normalized)
        self._gallery_lab: Dict[str, Tuple[float, float, float]] = {}  # gid -> LAB coords
    
    def _get_config(self):
        """Lazy load color config."""
        if self._color_config is None:
            self._color_config = get_color_config()
        return self._color_config
    
    def _normalize_color_name(self, name: str) -> str:
        """Normalize color name for lookup."""
        return (name or "").strip().lower()
    
    def _get_lab_for_color(self, color_name: str) -> Optional[Tuple[float, float, float]]:
        """Get LAB coordinates for a color name."""
        color_name = self._normalize_color_name(color_name)
        if not color_name:
            return None
        
        config = self._get_config()
        color_def = config.get_color(color_name)
        
        if color_def is not None:
            return color_def.lab
        
        # Color not in database - return None (will trigger fallback if enabled)
        return None
    
    def build_gallery(self, gallery_rows_by_id: Dict[str, Row]) -> None:
        """
        Index gallery colors and pre-compute LAB coordinates.
        """
        self._gallery_colors.clear()
        self._gallery_lab.clear()
        
        config = self._get_config()
        
        unknown_colors = set()
        
        for gid, row in gallery_rows_by_id.items():
            color_name = self._normalize_color_name(row.get(self.name, ""))
            if not color_name:
                continue
            
            self._gallery_colors[gid] = color_name
            
            # Get LAB coordinates
            lab = self._get_lab_for_color(color_name)
            if lab is not None:
                self._gallery_lab[gid] = lab
            else:
                unknown_colors.add(color_name)
        
        n_with_color = len(self._gallery_colors)
        n_with_lab = len(self._gallery_lab)
        
        _log.info(
            "build field=%s n_with_color=%d n_with_lab=%d threshold=%.1f",
            self.name, n_with_color, n_with_lab, self.threshold
        )
        
        if unknown_colors:
            _log.warning(
                "field=%s unknown_colors=%d: %s",
                self.name, len(unknown_colors), sorted(unknown_colors)[:10]
            )
    
    def prepare_query(self, q_row: Row) -> Any:
        """
        Prepare query state: extract color name and LAB coordinates.
        
        Returns dict with 'name' and 'lab' (lab may be None if unknown color).
        """
        color_name = self._normalize_color_name(q_row.get(self.name, ""))
        if not color_name:
            return None
        
        lab = self._get_lab_for_color(color_name)
        
        return {
            "name": color_name,
            "lab": lab,
        }
    
    def has_query_signal(self, q_state: Any) -> bool:
        """Returns True if the query has a color value."""
        return q_state is not None and q_state.get("name")
    
    def score_pair(self, q_state: Any, gallery_id: str) -> Tuple[float, bool]:
        """
        Score color similarity between query and gallery member.
        
        Returns:
            (score, present): score in [0,1], present=True if both have colors
        """
        if q_state is None:
            return 0.0, False
        
        q_name = q_state.get("name", "")
        q_lab = q_state.get("lab")
        
        # Check if gallery has this field
        g_name = self._gallery_colors.get(gallery_id, "")
        if not g_name:
            return 0.0, False
        
        # Exact name match = perfect score
        if q_name == g_name:
            return 1.0, True
        
        # Try LAB-based comparison
        g_lab = self._gallery_lab.get(gallery_id)
        
        if q_lab is not None and g_lab is not None:
            # Both colors have LAB coordinates - use perceptual distance
            delta_e = delta_e_cie2000(q_lab, g_lab)
            import math
            similarity = math.exp(-delta_e / self.threshold)
            return similarity, True
        
        # Fallback: one or both colors not in database
        if self.fallback_to_exact:
            # Exact match already handled above, so this is a mismatch
            return 0.0, True
        else:
            # Can't compare - treat as missing
            return 0.0, False


class ColorExactMatchScorer:
    """
    Simple exact-match scorer for colors (backward compatible).
    
    Use this if you want binary matching instead of perceptual similarity.
    """
    
    def __init__(self, field: str):
        self.name = field
        self.values_by_id: Dict[str, str] = {}
    
    def build_gallery(self, gallery_rows_by_id: Dict[str, Row]) -> None:
        self.values_by_id = {
            gid: (r.get(self.name, "") or "").strip().lower()
            for gid, r in gallery_rows_by_id.items()
        }
        n = sum(1 for v in self.values_by_id.values() if v)
        _log.info("build field=%s n_nonempty=%d (exact match)", self.name, n)
    
    def prepare_query(self, q_row: Row) -> Any:
        v = (q_row.get(self.name, "") or "").strip().lower()
        return v if v else None
    
    def has_query_signal(self, q_state: Any) -> bool:
        return q_state is not None
    
    def score_pair(self, q_state: Any, gallery_id: str) -> Tuple[float, bool]:
        v = self.values_by_id.get(gallery_id, "")
        if q_state is None or not v:
            return 0.0, False
        return (1.0 if v == q_state else 0.0), True



