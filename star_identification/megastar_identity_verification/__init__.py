"""
MegaStar Identity Verification

A pairwise identity verification system using cross-attention transformers.
Takes two images and outputs probability they show the same individual.

Two-stage re-identification pipeline:
1. Embedding model retrieves top-K candidates (fast)
2. Verification model confirms matches (accurate)

Compartmentalized from main codebase - minimal external dependencies.
"""

from .model import VerificationModel, CrossAttentionModule
from .config import VerificationConfig, TrainingConfig
from .inference import VerificationInference, load_inference

__all__ = [
    'VerificationModel',
    'CrossAttentionModule', 
    'VerificationConfig',
    'TrainingConfig',
    'VerificationInference',
    'load_inference',
]






