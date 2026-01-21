"""
MegaStarID Alignment Utilities

Tools for analyzing embedding distance sensitivity to input perturbations,
with explicit temporal (outing-to-outing) tracking.
"""

from .sensitivity_analyzer import SensitivityAnalyzer
from .temporal_sensitivity_analysis import (
    TemporalSensitivityAnalyzer,
    BatchSensitivityAnalyzer,  # Alias for backward compatibility
    OutingInfo,
    OutingPairResult,
    IdentityTemporalResults,
    TemporalAnalysisResults,
    PERTURBATION_CONFIGS,
    TIME_GAP_BUCKETS,
)

__all__ = [
    # Main analyzers
    "SensitivityAnalyzer",
    "TemporalSensitivityAnalyzer",
    "BatchSensitivityAnalyzer",
    
    # Data classes
    "OutingInfo",
    "OutingPairResult", 
    "IdentityTemporalResults",
    "TemporalAnalysisResults",
    
    # Configs
    "PERTURBATION_CONFIGS",
    "TIME_GAP_BUCKETS",
]
