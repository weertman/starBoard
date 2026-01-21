# src/search/fields_short_arm_code.py
"""
Fuzzy scorer for short_arm_code field using position-aware bipartite matching.

Short arm codes describe which arms are reduced and by how much:
- Position: arm number around the sea star (like a clock, relative to madreporite)
- Severity: tiny < small < short (ordinal scale)

This scorer:
1. Parses codes into {position: severity} dictionaries
2. Computes pairwise arm similarity using Gaussian decay on circular position distance
3. Uses optimal bipartite matching (Hungarian algorithm) to pair arms
4. Normalizes by the larger arm count to penalize unmatched arms
"""
from __future__ import annotations
from typing import Dict, Set, Any, Tuple, List, Optional
import re
import math
import logging

Row = Dict[str, str]
_log = logging.getLogger("starBoard.search.field.short_arm_code")

# Severity ordinal mapping
SEV_ORD: Dict[str, int] = {"very_tiny": 0, "tiny": 1, "small": 2, "short": 3}
SEV_NAMES: Dict[int, str] = {0: "very_tiny", 1: "tiny", 2: "small", 3: "short"}

# Pattern to parse "very_tiny(2)", "tiny(2)", "small(10)", etc.
_ARM_PATTERN = re.compile(r"(very_tiny|tiny|small|short)\s*\(\s*(\d+)\s*\)", re.IGNORECASE)


def parse_arm_codes(raw: str) -> Dict[int, str]:
    """
    Parse short arm code string into {position: severity}.
    E.g., "tiny(2), small(5)" → {2: "tiny", 5: "small"}
    
    If the same position appears multiple times, keeps the most severe.
    """
    result: Dict[int, str] = {}
    for match in _ARM_PATTERN.finditer(raw or ""):
        sev = match.group(1).lower()
        pos = int(match.group(2))
        # If same position appears multiple times, keep the most severe
        if pos not in result or SEV_ORD[sev] > SEV_ORD.get(result[pos], -1):
            result[pos] = sev
    return result


def circular_distance(p1: int, p2: int, n_arms: int) -> int:
    """
    Circular distance between two positions on an n-armed star.
    
    E.g., on a 5-armed star, positions 1 and 5 are distance 1 apart (not 4).
    """
    if n_arms <= 0:
        return abs(p1 - p2)
    diff = abs(p1 - p2)
    return min(diff, n_arms - diff)


def position_similarity(p1: int, p2: int, n_arms: int, sigma: float = 1.0) -> float:
    """
    Gaussian decay based on circular distance.
    
    With σ=1.0:
        dist 0 → 1.00 (exact match)
        dist 1 → 0.61 (adjacent)
        dist 2 → 0.14
        dist 3 → 0.01
    
    Args:
        p1, p2: Arm positions
        n_arms: Total number of arms (for circular distance)
        sigma: Gaussian decay parameter (higher = more tolerant)
    """
    dist = circular_distance(p1, p2, n_arms)
    return math.exp(-(dist ** 2) / (2 * sigma ** 2))


def severity_similarity(sev1: str, sev2: str) -> float:
    """
    Ordinal similarity between severities.
    
    Returns:
        1.0 if same severity
        0.67 if off by 1 (e.g., very_tiny vs tiny)
        0.33 if off by 2 (e.g., very_tiny vs small)
        0.0 if off by 3 (e.g., very_tiny vs short)
    """
    ord1 = SEV_ORD.get(sev1.lower() if sev1 else "", 0)
    ord2 = SEV_ORD.get(sev2.lower() if sev2 else "", 0)
    d = abs(ord1 - ord2)
    return max(0.0, 1.0 - d / 3.0)


def arm_pair_similarity(
    q_pos: int, q_sev: str, g_pos: int, g_sev: str, n_arms: int, sigma: float = 1.0
) -> float:
    """
    Combined similarity for a pair of arms (multiplicative).
    
    Both position and severity must be reasonably close for a high score.
    """
    pos_sim = position_similarity(q_pos, g_pos, n_arms, sigma)
    sev_sim = severity_similarity(q_sev, g_sev)
    return pos_sim * sev_sim


def _greedy_match(sim_matrix: List[List[float]]) -> List[Tuple[int, int]]:
    """
    Greedy bipartite matching: pick highest similarity pairs first.
    
    This is a fallback when scipy is not available. It's not guaranteed
    to find the optimal matching, but works well in practice for small sets.
    
    Returns:
        List of (row_idx, col_idx) pairs representing the matching.
    """
    if not sim_matrix or not sim_matrix[0]:
        return []
    
    n_rows = len(sim_matrix)
    n_cols = len(sim_matrix[0])
    
    # Flatten to (sim, row, col) and sort descending by similarity
    candidates: List[Tuple[float, int, int]] = []
    for i in range(n_rows):
        for j in range(n_cols):
            candidates.append((sim_matrix[i][j], i, j))
    candidates.sort(reverse=True, key=lambda x: x[0])
    
    matched_rows: Set[int] = set()
    matched_cols: Set[int] = set()
    matches: List[Tuple[int, int]] = []
    
    for sim, i, j in candidates:
        if i not in matched_rows and j not in matched_cols:
            matches.append((i, j))
            matched_rows.add(i)
            matched_cols.add(j)
            if len(matches) >= min(n_rows, n_cols):
                break
    
    return matches


def _hungarian_match(sim_matrix: List[List[float]]) -> List[Tuple[int, int]]:
    """
    Optimal bipartite matching using Hungarian algorithm (scipy).
    
    Falls back to greedy matching if scipy is unavailable.
    
    Returns:
        List of (row_idx, col_idx) pairs representing the optimal matching.
    """
    if not sim_matrix or not sim_matrix[0]:
        return []
    
    try:
        from scipy.optimize import linear_sum_assignment
        import numpy as np
        
        # Convert to cost matrix (Hungarian minimizes, we want to maximize similarity)
        cost = 1.0 - np.array(sim_matrix)
        row_ind, col_ind = linear_sum_assignment(cost)
        return list(zip(row_ind.tolist(), col_ind.tolist()))
    except ImportError:
        _log.debug("scipy not available, using greedy matching for short_arm_code")
        return _greedy_match(sim_matrix)


def score_arm_codes(
    q_codes: Dict[int, str],
    g_codes: Dict[int, str],
    n_arms: int = 5,
    sigma: float = 1.0,
) -> float:
    """
    Compute similarity between two short arm code dictionaries.
    
    Uses optimal bipartite matching to pair query arms with gallery arms,
    considering both position proximity (circular) and severity similarity.
    
    Args:
        q_codes: Query arm codes {position: severity}
        g_codes: Gallery arm codes {position: severity}
        n_arms: Total number of arms (for circular distance calculation)
        sigma: Gaussian decay parameter for position similarity
    
    Returns:
        Similarity score in [0, 1]
    """
    # Edge cases
    if not q_codes and not g_codes:
        return 1.0  # Both have no short arms → perfect match (both normal)
    if not q_codes or not g_codes:
        return 0.0  # One has short arms, other doesn't → mismatch
    
    q_items = list(q_codes.items())
    g_items = list(g_codes.items())
    n_q, n_g = len(q_items), len(g_items)
    
    # Build similarity matrix [n_q × n_g]
    sim_matrix: List[List[float]] = []
    for q_pos, q_sev in q_items:
        row: List[float] = []
        for g_pos, g_sev in g_items:
            sim = arm_pair_similarity(q_pos, q_sev, g_pos, g_sev, n_arms, sigma)
            row.append(sim)
        sim_matrix.append(row)
    
    # Find optimal matching
    matches = _hungarian_match(sim_matrix)
    
    # Sum matched similarities
    total_sim = sum(sim_matrix[i][j] for i, j in matches)
    
    # Normalize by max count (penalizes unmatched arms on either side)
    return total_sim / max(n_q, n_g)


class ShortArmCodeScorer:
    """
    Fuzzy scorer for short_arm_code field using position-aware bipartite matching.
    
    Implements the FieldScorer protocol from src/search/interfaces.py.
    
    Attributes:
        name: Field name (should be "short_arm_code")
        sigma: Gaussian decay parameter for position tolerance
        default_n_arms: Fallback arm count when not specified in metadata
    """
    
    def __init__(
        self,
        field: str = "short_arm_code",
        sigma: float = 1.0,
        default_n_arms: int = 5,
    ):
        """
        Initialize the scorer.
        
        Args:
            field: Field name in the CSV (default: "short_arm_code")
            sigma: Position tolerance. Higher = more forgiving of position shifts.
                   σ=0.5: strict, σ=1.0: moderate (recommended), σ=1.5: lenient
            default_n_arms: Fallback when num_total_arms isn't available
        """
        self.name = field
        self.sigma = sigma
        self.default_n_arms = default_n_arms
        self.codes_by_id: Dict[str, Dict[int, str]] = {}
        self.n_arms_by_id: Dict[str, int] = {}
    
    def build_gallery(self, gallery_rows_by_id: Dict[str, Row]) -> None:
        """
        Parse and index short arm codes for all gallery members.
        
        Also extracts num_total_arms (or num_apparent_arms as fallback)
        for each individual to use in circular distance calculations.
        """
        self.codes_by_id.clear()
        self.n_arms_by_id.clear()
        
        for gid, row in gallery_rows_by_id.items():
            # Parse the short arm codes
            codes = parse_arm_codes(row.get(self.name, ""))
            self.codes_by_id[gid] = codes
            
            # Try to get num_total_arms for this individual
            n_arms = self.default_n_arms
            for arm_field in ("num_total_arms", "num_apparent_arms"):
                try:
                    v = int(float(row.get(arm_field, "") or ""))
                    if v > 0:
                        n_arms = v
                        break
                except (ValueError, TypeError):
                    pass
            self.n_arms_by_id[gid] = n_arms
        
        # Log statistics
        nonempty = sum(1 for c in self.codes_by_id.values() if c)
        avg_count = (
            sum(len(c) for c in self.codes_by_id.values() if c) / nonempty
            if nonempty else 0.0
        )
        _log.info(
            "build field=%s n_nonempty=%d avg_arms_per_code=%.2f sigma=%.2f default_n_arms=%d",
            self.name, nonempty, avg_count, self.sigma, self.default_n_arms
        )
    
    def prepare_query(self, q_row: Row) -> Any:
        """
        Prepare query state: parse codes and determine arm count.
        
        Returns None if the query has no short arm codes (field is empty).
        """
        codes = parse_arm_codes(q_row.get(self.name, ""))
        
        # Get num_arms for query
        n_arms = self.default_n_arms
        for arm_field in ("num_total_arms", "num_apparent_arms"):
            try:
                v = int(float(q_row.get(arm_field, "") or ""))
                if v > 0:
                    n_arms = v
                    break
            except (ValueError, TypeError):
                pass
        
        # Return state dict (or None if no codes)
        # Note: we return state even if codes is empty, so we can detect
        # "query has no short arms" vs "query field is missing"
        return {"codes": codes, "n_arms": n_arms}
    
    def has_query_signal(self, q_state: Any) -> bool:
        """
        Returns True if the query has usable signal for this field.
        
        We consider the field "present" even if codes is empty (meaning
        the individual was annotated as having no short arms), as this
        is meaningful information for comparison.
        """
        # If q_state is None, the field wasn't even parsed (shouldn't happen)
        # If q_state exists, we have signal (even if codes dict is empty)
        return q_state is not None
    
    def score_pair(self, q_state: Any, gallery_id: str) -> Tuple[float, bool]:
        """
        Score similarity between query and a gallery member.
        
        Returns:
            (score, present): score in [0,1], present=True if both have the field
        """
        if q_state is None:
            return 0.0, False
        
        g_codes = self.codes_by_id.get(gallery_id)
        
        # If gallery member not in index, we can't compare
        if g_codes is None:
            return 0.0, False
        
        q_codes = q_state.get("codes", {})
        
        # Both empty = both normal = perfect match
        if not q_codes and not g_codes:
            return 1.0, True
        
        # One empty, one not = mismatch (but field is present)
        if not q_codes or not g_codes:
            return 0.0, True
        
        # Use the larger n_arms between query and gallery for distance calc
        # (handles cases where one has regenerated arms)
        q_n_arms = q_state.get("n_arms", self.default_n_arms)
        g_n_arms = self.n_arms_by_id.get(gallery_id, self.default_n_arms)
        n_arms = max(q_n_arms, g_n_arms)
        
        score = score_arm_codes(q_codes, g_codes, n_arms=n_arms, sigma=self.sigma)
        return score, True



