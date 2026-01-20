"""
Pair Dataset for verification training.

Samples balanced positive (same identity) and negative (different identity) pairs.
"""
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class PairDataset(Dataset):
    """
    Dataset that yields pairs of images for verification training.
    
    Samples a mix of positive pairs (same identity) and negative pairs
    (different identities) each epoch.
    """
    
    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        transform=None,
        pairs_per_epoch: int = 50000,
        positive_ratio: float = 0.5,
        seed: int = 42,
    ):
        """
        Args:
            image_paths: List of paths to images
            labels: List of identity labels (same length as image_paths)
            transform: Image transform to apply
            pairs_per_epoch: Number of pairs to sample per epoch
            positive_ratio: Fraction of positive pairs (0.5 = balanced)
            seed: Random seed for reproducibility
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.pairs_per_epoch = pairs_per_epoch
        self.positive_ratio = positive_ratio
        
        # Build label -> indices mapping
        self.label_to_indices: Dict[int, List[int]] = defaultdict(list)
        for idx, label in enumerate(labels):
            self.label_to_indices[label].append(idx)
        
        # Filter to identities with at least 2 images (for positive pairs)
        self.valid_labels = [
            label for label, indices in self.label_to_indices.items()
            if len(indices) >= 2
        ]
        self.all_labels = list(self.label_to_indices.keys())
        
        print(f"PairDataset: {len(image_paths)} images, {len(self.all_labels)} identities")
        print(f"  {len(self.valid_labels)} identities have >=2 images (for positive pairs)")
        print(f"  Sampling {pairs_per_epoch} pairs per epoch ({positive_ratio:.0%} positive)")
        
        # Random state
        self.rng = np.random.RandomState(seed)
        
        # Pre-sample pairs for this epoch
        self._resample_pairs()
    
    def _resample_pairs(self):
        """Resample pairs for a new epoch."""
        num_pos = int(self.pairs_per_epoch * self.positive_ratio)
        num_neg = self.pairs_per_epoch - num_pos
        
        pairs = []
        
        # Sample positive pairs
        for _ in range(num_pos):
            # Pick identity with at least 2 images
            label = self.rng.choice(self.valid_labels)
            indices = self.label_to_indices[label]
            # Pick 2 different images
            idx_a, idx_b = self.rng.choice(indices, size=2, replace=False)
            pairs.append((idx_a, idx_b, 1))  # 1 = same
        
        # Sample negative pairs
        for _ in range(num_neg):
            # Pick 2 different identities
            label_a, label_b = self.rng.choice(self.all_labels, size=2, replace=False)
            # Pick 1 image from each
            idx_a = self.rng.choice(self.label_to_indices[label_a])
            idx_b = self.rng.choice(self.label_to_indices[label_b])
            pairs.append((idx_a, idx_b, 0))  # 0 = different
        
        # Shuffle
        self.rng.shuffle(pairs)
        self.pairs = pairs
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        idx_a, idx_b, label = self.pairs[idx]
        
        # Load images
        img_a = Image.open(self.image_paths[idx_a]).convert('RGB')
        img_b = Image.open(self.image_paths[idx_b]).convert('RGB')
        
        # Apply transform
        if self.transform is not None:
            img_a = self.transform(img_a)
            img_b = self.transform(img_b)
        
        return {
            'image_a': img_a,
            'image_b': img_b,
            'label': torch.tensor(label, dtype=torch.float32),
            'identity_a': self.labels[idx_a],
            'identity_b': self.labels[idx_b],
        }
    
    def on_epoch_end(self):
        """Call at end of epoch to resample pairs."""
        self._resample_pairs()


class Wildlife10kPairDataset(PairDataset):
    """
    Pair dataset specifically for Wildlife10k.
    
    Wraps the Wildlife10k metadata to create verification pairs.
    """
    
    @classmethod
    def from_wildlife10k(
        cls,
        data_root: str,
        mode: str = 'train',
        split_strategy: str = 'recommended',
        exclude_datasets: Optional[List[str]] = None,
        transform=None,
        pairs_per_epoch: int = 50000,
        positive_ratio: float = 0.5,
        seed: int = 42,
    ) -> 'Wildlife10kPairDataset':
        """
        Create pair dataset from Wildlife10k.
        
        Args:
            data_root: Path to Wildlife10k data
            mode: 'train' or 'test'
            split_strategy: How to split train/test
            exclude_datasets: Datasets to exclude (e.g., ["SeaStarReID2023"])
            transform: Image transform
            pairs_per_epoch: Pairs to sample per epoch
            positive_ratio: Fraction of positive pairs
            seed: Random seed
        """
        # Import here to avoid circular imports
        from megastarid.datasets.combined import (
            Wildlife10kDataset, Wildlife10kConfig, FilterConfig, SplitConfig
        )
        
        # Create Wildlife10k config
        config = Wildlife10kConfig(
            data_root=data_root,
            filter=FilterConfig(
                include_datasets=[],
                exclude_datasets=exclude_datasets or [],
            ),
            split=SplitConfig(strategy=split_strategy),
        )
        
        # Create dataset to get image paths and labels
        wildlife_dataset = Wildlife10kDataset(
            data_root=data_root,
            transform=None,  # We'll apply transform ourselves
            mode=mode,
            config=config,
        )
        
        # Extract paths and labels from the DataFrame
        # Wildlife10kDataset stores data in self.df with 'path' and 'identity' columns
        image_paths = wildlife_dataset.df['path'].tolist()
        labels = [
            wildlife_dataset.identity_to_label[identity] 
            for identity in wildlife_dataset.df['identity']
        ]
        
        return cls(
            image_paths=image_paths,
            labels=labels,
            transform=transform,
            pairs_per_epoch=pairs_per_epoch,
            positive_ratio=positive_ratio,
            seed=seed,
        )


class StarDatasetPairDataset(PairDataset):
    """
    Pair dataset for star_dataset.
    
    Uses temporal splits from metadata files for realistic evaluation.
    """
    
    @classmethod
    def from_star_dataset(
        cls,
        data_root: str,
        mode: str = 'train',
        transform=None,
        pairs_per_epoch: int = 10000,
        positive_ratio: float = 0.5,
        seed: int = 42,
        train_outing_ratio: float = 0.8,
        min_outings_for_eval: int = 2,
        include_inaturalist: bool = False,
        include_negative_only: bool = False,
    ) -> 'StarDatasetPairDataset':
        """
        Create pair dataset from star_dataset using metadata files.
        
        Args:
            data_root: Path to star_dataset
            mode: 'train' or 'test'
            transform: Image transform
            pairs_per_epoch: Pairs to sample per epoch
            positive_ratio: Fraction of positive pairs
            seed: Random seed
            train_outing_ratio: Fraction of outings for training
            min_outings_for_eval: Minimum outings needed for train/test split
            include_inaturalist: Whether to include iNaturalist data
            include_negative_only: Whether to include negative-only identities
                (single-outing identities that can only form negative pairs)
        """
        # Import here to avoid circular imports
        from megastarid.datasets.combined import StarDataset
        
        # Create StarDataset which handles metadata loading and temporal splitting
        star_dataset = StarDataset(
            data_root=data_root,
            transform=None,  # We'll apply transform ourselves
            mode=mode,
            train_outing_ratio=train_outing_ratio,
            min_outings_for_eval=min_outings_for_eval,
            seed=seed,
            include_inaturalist=include_inaturalist,
        )
        
        # Get set of negative-only labels to filter if needed
        negative_only_labels = star_dataset.negative_only_labels
        
        # Extract paths and labels from the dataset
        image_paths = []
        labels = []
        n_excluded = 0
        
        for idx in range(len(star_dataset)):
            row = star_dataset.df.iloc[idx]
            label = star_dataset.identity_to_label[row['identity']]
            
            # Skip negative-only identities if not included
            if not include_negative_only and label in negative_only_labels:
                n_excluded += 1
                continue
            
            # Build path
            if 'path' in row and pd.notna(row['path']):
                path = row['path']
                if not Path(path).is_absolute():
                    path = str(Path(data_root) / path)
            else:
                path = str(Path(data_root) / row['folder'] / row['filename'])
            
            image_paths.append(path)
            labels.append(label)
        
        n_identities = len(set(labels))
        print(f"  Loaded {len(image_paths)} images, {n_identities} identities in {mode} split")
        if n_excluded > 0:
            print(f"  Excluded {n_excluded} images from {len(negative_only_labels)} negative-only identities")
        
        return cls(
            image_paths=image_paths,
            labels=labels,
            transform=transform,
            pairs_per_epoch=pairs_per_epoch,
            positive_ratio=positive_ratio,
            seed=seed,
        )


def create_pair_dataloaders(
    dataset: PairDataset,
    batch_size: int = 32,
    num_workers: int = 8,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Create DataLoader for pair dataset.
    
    Note: Don't use shuffle=True since pairs are pre-sampled and shuffled.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Already shuffled in dataset
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,  # For consistent batch sizes
    )

