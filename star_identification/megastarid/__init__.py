"""
MegaStarID - Unified Training for Sea Star Re-Identification

Integrates:
- Wildlife10k (140k images, 37 species) for broad visual feature learning
- star_dataset (8k images, sea stars) for domain-specific fine-tuning

Training Strategies:
1. Pre-train → Fine-tune: Train on Wildlife10k, then fine-tune on star_dataset
2. Co-training: Train on both datasets simultaneously with mixed batches

Usage:
    # Pre-training on Wildlife10k (excludes sea stars for fair eval)
    python -m megastarid.pretrain --epochs 50
    
    # Fine-tuning on star_dataset
    python -m megastarid.finetune --checkpoint checkpoints/megastarid/pretrain_best.pth
    
    # Co-training on both
    python -m megastarid.cotrain --epochs 100 --star-weight 0.3
"""

from .config import MegaStarConfig, PretrainConfig, FinetuneConfig, CotrainConfig
from .evaluation import compute_detailed_star_metrics, evaluate_star_dataset

__all__ = [
    'MegaStarConfig',
    'PretrainConfig', 
    'FinetuneConfig',
    'CotrainConfig',
    'compute_detailed_star_metrics',
    'evaluate_star_dataset',
]

