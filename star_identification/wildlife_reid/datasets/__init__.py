"""
Dataset classes for Wildlife10k.
"""
from .base import BaseWildlifeDataset
from .wildlife10k import Wildlife10kDataset
from .subdataset import SubDatasetHandler

__all__ = [
    'BaseWildlifeDataset',
    'Wildlife10kDataset', 
    'SubDatasetHandler',
]


