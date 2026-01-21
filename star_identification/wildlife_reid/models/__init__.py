"""
Model definitions for Wildlife ReID.

Includes:
- SwinReID: Original Swin-based model (legacy)
- MultiScaleReID: New multi-scale model with multiple backbone options
"""
from .swin_reid import SwinReID, GeM, create_model
from .multiscale_reid import (
    MultiScaleReID,
    create_multiscale_model,
    GeM as GeM2,  # Updated GeM with 4D support
    AttentionPooling,
    MultiScaleFeatureFusion,
    EmbeddingHead,
)

__all__ = [
    # Legacy
    'SwinReID',
    'GeM',
    'create_model',
    # New multi-scale
    'MultiScaleReID',
    'create_multiscale_model',
    'AttentionPooling',
    'MultiScaleFeatureFusion',
    'EmbeddingHead',
]

