"""
Wildlife ReID-10k Dataset Module

A modular, dataset-aware handling system for the WildlifeReID-10k benchmark.
Designed for seamless integration with temporal_reid training pipelines.

Key Features:
- Per-dataset split logic (time-aware, cluster-aware, or random)
- Registry of 37 sub-datasets with their characteristics
- Unified PyTorch Dataset interface
- Compatible with P-K sampling for metric learning
"""

from .registry import DatasetRegistry, DATASET_REGISTRY
from .loader import Wildlife10kLoader
from .config import Wildlife10kConfig, FilterConfig, SplitConfig

__all__ = [
    'DatasetRegistry',
    'DATASET_REGISTRY', 
    'Wildlife10kLoader',
    'Wildlife10kConfig',
    'FilterConfig',
    'SplitConfig',
]

