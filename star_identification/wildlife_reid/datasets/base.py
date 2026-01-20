"""
Base dataset class for Wildlife ReID.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image


class BaseWildlifeDataset(Dataset, ABC):
    """
    Abstract base class for wildlife re-identification datasets.
    
    Provides common functionality for:
    - Image loading with error handling
    - Identity-to-label mapping
    - P-K sampling support via label_to_indices
    - Transform application
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        transform=None,
        mode: str = 'train',
    ):
        """
        Args:
            df: DataFrame with columns [identity, path, split, ...]
            transform: Torchvision transforms to apply
            mode: 'train' or 'test'
        """
        self.mode = mode
        self.transform = transform
        
        # Filter by split
        self.df = df[df['split'] == mode].reset_index(drop=True)
        
        if len(self.df) == 0:
            raise ValueError(f"No samples found for mode='{mode}'")
        
        # Create identity to integer label mapping
        self.identities = sorted(self.df['identity'].unique())
        self.identity_to_label = {
            identity: idx for idx, identity in enumerate(self.identities)
        }
        self.label_to_identity = {
            idx: identity for identity, idx in self.identity_to_label.items()
        }
        
        # Build label_to_indices for P-K sampling
        self.label_to_indices: Dict[int, List[int]] = defaultdict(list)
        for idx, row in self.df.iterrows():
            label = self.identity_to_label[row['identity']]
            self.label_to_indices[label].append(idx)
        self.label_to_indices = dict(self.label_to_indices)
        
        # Statistics
        self.num_identities = len(self.identities)
        self.num_images = len(self.df)
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample."""
        row = self.df.iloc[idx]
        
        # Load image
        image = self._load_image(row['path'])
        
        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)
        
        # Get label
        label = self.identity_to_label[row['identity']]
        
        return {
            'image': image,
            'label': label,
            'identity': row['identity'],
            'path': row['path'],
            'dataset': row.get('dataset', 'unknown'),
            'species': row.get('species', 'unknown'),
        }
    
    def _load_image(self, path: str) -> Image.Image:
        """Load image with error handling."""
        try:
            img = Image.open(path).convert('RGB')
            return img
        except Exception as e:
            print(f"Error loading {path}: {e}")
            # Return a small black image as fallback
            return Image.new('RGB', (224, 224), (0, 0, 0))
    
    def get_identities_with_min_samples(self, min_samples: int) -> List[str]:
        """Get identities that have at least min_samples images."""
        counts = self.df['identity'].value_counts()
        return counts[counts >= min_samples].index.tolist()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        stats = {
            'mode': self.mode,
            'num_images': self.num_images,
            'num_identities': self.num_identities,
            'images_per_identity': {
                'mean': self.df['identity'].value_counts().mean(),
                'min': self.df['identity'].value_counts().min(),
                'max': self.df['identity'].value_counts().max(),
            }
        }
        
        # Per-dataset breakdown if available
        if 'dataset' in self.df.columns:
            dataset_stats = {}
            for ds in self.df['dataset'].unique():
                ds_df = self.df[self.df['dataset'] == ds]
                dataset_stats[ds] = {
                    'images': len(ds_df),
                    'identities': ds_df['identity'].nunique(),
                }
            stats['per_dataset'] = dataset_stats
        
        # Per-species breakdown if available
        if 'species' in self.df.columns:
            species_stats = {}
            for sp in self.df['species'].unique():
                sp_df = self.df[self.df['species'] == sp]
                species_stats[sp] = {
                    'images': len(sp_df),
                    'identities': sp_df['identity'].nunique(),
                }
            stats['per_species'] = species_stats
        
        return stats
    
    def print_summary(self):
        """Print a summary of the dataset."""
        stats = self.get_statistics()
        print(f"\n{self.mode.upper()} Dataset Summary")
        print("=" * 40)
        print(f"Images: {stats['num_images']}")
        print(f"Identities: {stats['num_identities']}")
        print(f"Images/identity: {stats['images_per_identity']['mean']:.1f} "
              f"(min={stats['images_per_identity']['min']}, "
              f"max={stats['images_per_identity']['max']})")
        
        if 'per_dataset' in stats:
            print(f"\nPer-dataset breakdown:")
            for ds, ds_stats in sorted(stats['per_dataset'].items()):
                print(f"  {ds}: {ds_stats['images']} images, {ds_stats['identities']} ids")


