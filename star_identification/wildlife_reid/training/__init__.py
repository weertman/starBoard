"""
Training utilities for Wildlife ReID.
"""
from .losses import CircleLoss, TripletLoss, CombinedLoss
from .trainer import Trainer

__all__ = ['CircleLoss', 'TripletLoss', 'CombinedLoss', 'Trainer']


