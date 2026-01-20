"""
Dataset and DataLoader with support for temporal splitting and negative-only identities.
"""
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Sampler


class TemporalReIDDataset(Dataset):
    """
    Dataset for temporal re-identification.
    
    Handles:
    - Multi-outing identities (evaluable, can be anchor/positive/negative)
    - Single-outing identities (negative-only, cannot be anchor/positive)
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        transform=None,
        mode: str = 'train',
        name: str = None,
    ):
        """
        Args:
            df: DataFrame with columns [identity, path, split, negative_only, ...]
            transform: Image transform
            mode: 'train' or 'test'
            name: Display name for logging (default: mode.upper())
        """
        self.mode = mode
        self.name = name or mode.upper()
        self.transform = transform
        
        # Filter to requested split
        self.df = df[df['split'] == mode].reset_index(drop=True)
        
        if len(self.df) == 0:
            raise ValueError(f"No samples found for split '{mode}'")
        
        # Build identity -> label mapping
        # All identities get labels (needed for batch construction)
        all_identities = sorted(df['identity'].unique())
        self.identity_to_label = {id_: i for i, id_ in enumerate(all_identities)}
        self.label_to_identity = {i: id_ for id_, i in self.identity_to_label.items()}
        self.num_classes = len(all_identities)
        
        # Track which identities are negative-only
        negative_only_df = df[df['negative_only'] == True]
        self.negative_only_labels = set(
            self.identity_to_label[id_] for id_ in negative_only_df['identity'].unique()
        )
        
        # Build index structures for sampling
        self.label_to_indices = defaultdict(list)
        self.evaluable_label_to_indices = defaultdict(list)  # Excludes negative-only
        
        for idx, row in self.df.iterrows():
            label = self.identity_to_label[row['identity']]
            self.label_to_indices[label].append(idx)
            
            if label not in self.negative_only_labels:
                self.evaluable_label_to_indices[label].append(idx)
        
        # Statistics
        n_evaluable = len(self.evaluable_label_to_indices)
        n_negative_only = len(self.negative_only_labels & set(self.label_to_indices.keys()))
        
        print(f"\n{self.name} Dataset:")
        print(f"  Total images: {len(self.df)}")
        print(f"  Total identities in split: {len(self.label_to_indices)}")
        print(f"  Evaluable (can be anchor): {n_evaluable}")
        print(f"  Negative-only: {n_negative_only}")
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]
        
        # Load image
        try:
            image = Image.open(row['path']).convert('RGB')
        except Exception as e:
            print(f"Error loading {row['path']}: {e}")
            image = Image.new('RGB', (224, 224), (128, 128, 128))
        
        if self.transform:
            image = self.transform(image)
        
        label = self.identity_to_label[row['identity']]
        negative_only = row.get('negative_only', False)
        
        return {
            'image': image,
            'label': label,
            'negative_only': negative_only,
            'identity': row['identity'],
            'path': row['path'],
        }
    
    def get_evaluable_labels(self) -> List[int]:
        """Get list of labels that can be used as anchors (not negative-only)."""
        return list(self.evaluable_label_to_indices.keys())
    
    def get_negative_only_labels(self) -> List[int]:
        """Get list of negative-only labels."""
        return list(self.negative_only_labels & set(self.label_to_indices.keys()))


class TemporalPKSampler(Sampler):
    """
    P-K sampler that respects negative-only constraints.
    
    - Anchors/Positives: only from evaluable (multi-outing) identities
    - Negatives: can include negative-only (single-outing) identities
    
    Each batch contains P identities with K instances each.
    At least some identities in each batch are evaluable.
    """
    
    def __init__(
        self,
        dataset: TemporalReIDDataset,
        batch_size: int,
        num_instances: int,
        evaluable_ratio: float = 0.75,  # Fraction of batch from evaluable identities
    ):
        """
        Args:
            dataset: TemporalReIDDataset
            batch_size: Total batch size (should be divisible by num_instances)
            num_instances: K - instances per identity
            evaluable_ratio: Fraction of identities in batch that should be evaluable
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_identities = batch_size // num_instances
        self.evaluable_ratio = evaluable_ratio
        
        # Separate evaluable and negative-only identities with enough samples
        self.evaluable_labels = []
        self.negative_only_labels = []
        
        for label, indices in dataset.label_to_indices.items():
            if len(indices) >= num_instances:
                if label in dataset.negative_only_labels:
                    self.negative_only_labels.append(label)
                else:
                    self.evaluable_labels.append(label)
        
        # Calculate how many of each type per batch
        self.num_evaluable_per_batch = max(2, int(self.num_identities * evaluable_ratio))
        self.num_negative_only_per_batch = self.num_identities - self.num_evaluable_per_batch
        
        # Ensure we have enough evaluable identities
        if len(self.evaluable_labels) < self.num_evaluable_per_batch:
            raise ValueError(
                f"Need at least {self.num_evaluable_per_batch} evaluable identities with "
                f">= {num_instances} samples, but only have {len(self.evaluable_labels)}"
            )
        
        # Calculate epoch length
        self._length = self._calculate_length()
        
        print(f"\nPK Sampler:")
        print(f"  Evaluable identities (can be anchor): {len(self.evaluable_labels)}")
        print(f"  Negative-only identities: {len(self.negative_only_labels)}")
        print(f"  Batch composition: {self.num_evaluable_per_batch} evaluable + "
              f"{self.num_negative_only_per_batch} negative-only")
        print(f"  Batches per epoch: {self._length}")
    
    def _calculate_length(self) -> int:
        """Calculate number of batches per epoch."""
        # Base on evaluable identities (they're the constraint)
        total_evaluable_samples = sum(
            len(self.dataset.label_to_indices[label]) 
            for label in self.evaluable_labels
        )
        samples_per_batch = self.num_evaluable_per_batch * self.num_instances
        return max(1, total_evaluable_samples // samples_per_batch)
    
    def __iter__(self):
        """Generate batches."""
        # Shuffle identity lists
        evaluable = self.evaluable_labels.copy()
        negative_only = self.negative_only_labels.copy()
        random.shuffle(evaluable)
        random.shuffle(negative_only)
        
        # Create index pools for each identity
        evaluable_pools = {
            label: list(self.dataset.label_to_indices[label]) 
            for label in evaluable
        }
        negative_only_pools = {
            label: list(self.dataset.label_to_indices[label])
            for label in negative_only
        }
        
        # Shuffle pools
        for pool in evaluable_pools.values():
            random.shuffle(pool)
        for pool in negative_only_pools.values():
            random.shuffle(pool)
        
        batches_yielded = 0
        evaluable_idx = 0
        negative_only_idx = 0
        
        while batches_yielded < self._length:
            batch = []
            
            # Sample evaluable identities
            batch_evaluable_labels = []
            for _ in range(self.num_evaluable_per_batch):
                # Cycle through evaluable labels
                label = evaluable[evaluable_idx % len(evaluable)]
                evaluable_idx += 1
                batch_evaluable_labels.append(label)
                
                # Sample instances
                pool = evaluable_pools[label]
                if len(pool) < self.num_instances:
                    # Refill pool
                    pool.extend(self.dataset.label_to_indices[label])
                    random.shuffle(pool)
                
                indices = [pool.pop() for _ in range(self.num_instances)]
                batch.extend(indices)
            
            # Sample negative-only identities (if available)
            if negative_only and self.num_negative_only_per_batch > 0:
                for _ in range(self.num_negative_only_per_batch):
                    label = negative_only[negative_only_idx % len(negative_only)]
                    negative_only_idx += 1
                    
                    pool = negative_only_pools[label]
                    if len(pool) < self.num_instances:
                        pool.extend(self.dataset.label_to_indices[label])
                        random.shuffle(pool)
                    
                    indices = [pool.pop() for _ in range(self.num_instances)]
                    batch.extend(indices)
            
            # Shuffle batch to mix evaluable and negative-only
            random.shuffle(batch)
            
            yield batch
            batches_yielded += 1
    
    def __len__(self) -> int:
        return self._length


def create_dataloaders(
    df: pd.DataFrame,
    config,
    train_transform,
    val_transform,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation (gallery), and query dataloaders.
    
    Args:
        df: Full metadata DataFrame with temporal splits
        config: Config object
        train_transform: Transform for training
        val_transform: Transform for validation
    
    Returns:
        train_loader: For training (P-K sampling)
        gallery_loader: Gallery images (train outings of evaluable identities)
        query_loader: Query images (test outings of evaluable identities)
    """
    # Create datasets
    train_dataset = TemporalReIDDataset(df, train_transform, mode='train')
    
    # For validation: only evaluable identities, test split
    evaluable_test_df = df[(df['split'] == 'test') & (df['negative_only'] == False)]
    
    if len(evaluable_test_df) == 0:
        print("WARNING: No evaluable test samples! Creating dummy test loader.")
        query_dataset = train_dataset  # Fallback
    else:
        query_dataset = TemporalReIDDataset(
            df[df['negative_only'] == False],  # Only evaluable identities
            val_transform, 
            mode='test',
            name='QUERY',
        )
    
    # Gallery: train images from evaluable identities
    gallery_dataset = TemporalReIDDataset(
        df[df['negative_only'] == False],
        val_transform,
        mode='train',
        name='GALLERY',
    )
    
    # Create samplers
    train_sampler = TemporalPKSampler(
        train_dataset,
        batch_size=config.batch_size,
        num_instances=config.num_instances,
    )
    
    # DataLoader settings
    loader_kwargs = {
        'num_workers': config.num_workers,
        'pin_memory': config.pin_memory,
    }
    
    if config.num_workers > 0:
        loader_kwargs['persistent_workers'] = True
        loader_kwargs['prefetch_factor'] = 2
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        **loader_kwargs
    )
    
    gallery_loader = DataLoader(
        gallery_dataset,
        batch_size=config.batch_size * 2,
        shuffle=False,
        **loader_kwargs
    )
    
    query_loader = DataLoader(
        query_dataset,
        batch_size=config.batch_size * 2,
        shuffle=False,
        **loader_kwargs
    )
    
    print(f"\nDataLoaders created:")
    print(f"  Train: {len(train_loader)} batches")
    print(f"  Gallery: {len(gallery_loader)} batches, {len(gallery_dataset)} images")
    print(f"  Query: {len(query_loader)} batches, {len(query_dataset)} images")
    
    return train_loader, gallery_loader, query_loader


