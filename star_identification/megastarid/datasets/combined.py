"""
Combined dataset handling for Wildlife10k and star_dataset.
"""
import platform
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Sampler

# Windows multiprocessing is slower and can deadlock with persistent_workers
IS_WINDOWS = platform.system() == 'Windows'

# Import from sibling modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from wildlife_reid.datasets import Wildlife10kDataset
from wildlife_reid.config import Wildlife10kConfig, FilterConfig, SplitConfig

# Import transforms from megastarid (supports both wildlife and star augmentations)
from megastarid.transforms import (
    get_wildlife_train_transforms,
    get_wildlife_test_transforms,
    get_star_train_transforms,
    get_star_test_transforms,
)


def get_eval_batch_multiplier(backbone: str) -> float:
    """
    Get eval batch size multiplier based on architecture.
    
    Transformers (SwinV2) still use significant memory during inference
    due to attention computation, so we can't scale up as aggressively
    as we can with CNNs (DenseNet).
    
    Args:
        backbone: Model backbone name ('swinv2-tiny', 'densenet121', etc.)
        
    Returns:
        Multiplier to apply to training batch size for eval
    """
    # Transformers: attention requires O(N²) memory even without gradients
    # Use conservative multiplier to avoid OOM during validation
    if 'swin' in backbone.lower() or 'vit' in backbone.lower():
        return 1.5  # Conservative for transformers
    
    # CNNs: constant memory during inference, can scale up more
    return 4.0  # Original multiplier for CNNs


class StarDataset(Dataset):
    """
    Dataset for star_dataset with temporal splitting support.
    
    Simplified version of TemporalReIDDataset for integration.
    
    IMPORTANT: The identity_to_label mapping is built from ALL identities
    in the dataset (before filtering by split) to ensure consistent labels
    between gallery and query datasets during evaluation.
    """
    
    def __init__(
        self,
        data_root: str,
        transform=None,
        mode: str = 'train',
        train_outing_ratio: float = 0.8,
        min_outings_for_eval: int = 2,
        seed: int = 42,
        label_offset: int = 0,  # Offset for label indices to avoid collision
    ):
        self.data_root = Path(data_root)
        self.transform = transform
        self.mode = mode
        self.label_offset = label_offset
        
        # Load metadata
        metadata_path = self.data_root / 'metadata_temporal.csv'
        if not metadata_path.exists():
            metadata_path = self.data_root / 'metadata.csv'
        
        df_full = pd.read_csv(metadata_path)
        
        # Apply temporal splitting if not already split
        if 'split' not in df_full.columns:
            df_full = self._apply_temporal_split(df_full, train_outing_ratio, min_outings_for_eval, seed)
        
        # BUILD IDENTITY MAPPING FROM ALL IDENTITIES (BEFORE FILTERING!)
        # This ensures gallery and query datasets have consistent label mappings
        all_identities_global = sorted(df_full['identity'].unique())
        self.identity_to_label = {id_: i + label_offset for i, id_ in enumerate(all_identities_global)}
        self.label_to_identity = {v: k for k, v in self.identity_to_label.items()}
        
        # Track negative-only labels (before filtering, so we know the full set)
        self.negative_only_labels = set()
        if 'negative_only' in df_full.columns:
            for identity in df_full[df_full['negative_only'] == True]['identity'].unique():
                if identity in self.identity_to_label:
                    self.negative_only_labels.add(self.identity_to_label[identity])
        
        # NOW filter to the requested mode (train/test)
        self.df = df_full[df_full['split'] == mode].reset_index(drop=True)
        
        # Build label_to_indices for THIS split only (for sampling)
        self.label_to_indices: Dict[int, List[int]] = defaultdict(list)
        for idx in range(len(self.df)):
            row = self.df.iloc[idx]
            label = self.identity_to_label[row['identity']]
            self.label_to_indices[label].append(idx)
        self.label_to_indices = dict(self.label_to_indices)
        
        # Count identities in this split (for summary)
        self.num_identities_global = len(all_identities_global)
        self.num_identities = len(self.label_to_indices)  # Identities in this split
        self.num_images = len(self.df)
    
    def _apply_temporal_split(
        self,
        df_input: pd.DataFrame,
        train_ratio: float,
        min_outings: int,
        seed: int,
    ) -> pd.DataFrame:
        """Apply temporal split based on outings."""
        df = df_input.copy()
        
        # Validate train_ratio to prevent edge cases
        if not (0.0 < train_ratio < 1.0):
            raise ValueError(
                f"train_ratio must be between 0 and 1 (exclusive), got {train_ratio}. "
                f"Use 0.8 for 80% train / 20% test split."
            )
        
        # Use a local RNG for determinism (avoids global state issues)
        rng = np.random.default_rng(seed)
        
        # Default split
        df['split'] = 'train'
        df['negative_only'] = False
        
        for identity in df['identity'].unique():
            mask = df['identity'] == identity
            id_df = df[mask]
            
            # Count unique outings (by date or folder)
            if 'date' in id_df.columns:
                outings = id_df['date'].nunique()
            elif 'folder' in id_df.columns:
                outings = id_df['folder'].nunique()
            else:
                outings = 1
            
            if outings >= min_outings:
                # Multi-outing: temporal split
                n_samples = len(id_df)
                n_train = int(n_samples * train_ratio)
                
                # Edge case: ensure both train and test get at least 1 sample
                if n_train == 0:
                    n_train = 1  # Ensure at least 1 in train (gallery)
                if n_train >= n_samples:
                    n_train = n_samples - 1  # Ensure at least 1 in test (query)
                
                indices = id_df.index.tolist()
                
                # If we have dates, sort by date
                if 'date' in id_df.columns:
                    id_df_sorted = id_df.sort_values('date')
                    indices = id_df_sorted.index.tolist()
                
                train_indices = indices[:n_train]
                test_indices = indices[n_train:]
                
                df.loc[train_indices, 'split'] = 'train'
                df.loc[test_indices, 'split'] = 'test'
            else:
                # Single-outing: train only, negative-only
                df.loc[mask, 'split'] = 'train'
                df.loc[mask, 'negative_only'] = True
        
        return df
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        
        # Build path
        if 'path' in row and pd.notna(row['path']):
            path = row['path']
            if not Path(path).is_absolute():
                path = self.data_root / path
        else:
            # Construct from folder/filename
            path = self.data_root / row['folder'] / row['filename']
        
        # Load image
        try:
            image = Image.open(path).convert('RGB')
        except Exception as e:
            print(f"Error loading {path}: {e}")
            image = Image.new('RGB', (384, 384), (128, 128, 128))
        
        if self.transform:
            image = self.transform(image)
        
        label = self.identity_to_label[row['identity']]
        
        return {
            'image': image,
            'label': label,
            'identity': row['identity'],
            'path': str(path),
            'dataset': 'star_dataset',
            'species': 'sea_star',
        }
    
    def print_summary(self):
        """Print dataset summary."""
        print(f"\nSTAR DATASET ({self.mode.upper()}):")
        print(f"  Images: {self.num_images}")
        print(f"  Identities in split: {self.num_identities}")
        print(f"  Identities total: {self.num_identities_global}")
        print(f"  Negative-only identities: {len(self.negative_only_labels)}")


class CombinedDataset(Dataset):
    """
    Combined dataset that wraps Wildlife10k and StarDataset.
    
    Can be used for:
    - Pre-training (Wildlife10k only)
    - Fine-tuning (star_dataset only)
    - Co-training (both mixed)
    """
    
    def __init__(
        self,
        wildlife_dataset: Optional[Wildlife10kDataset] = None,
        star_dataset: Optional[StarDataset] = None,
        mode: str = 'train',
    ):
        self.wildlife_dataset = wildlife_dataset
        self.star_dataset = star_dataset
        self.mode = mode
        
        # Calculate offsets and combined structures
        self.wildlife_offset = 0
        self.star_offset = 0
        
        if wildlife_dataset is not None:
            self.wildlife_len = len(wildlife_dataset)
            self.star_offset = self.wildlife_len
        else:
            self.wildlife_len = 0
        
        if star_dataset is not None:
            self.star_len = len(star_dataset)
        else:
            self.star_len = 0
        
        # Build combined label_to_indices
        self.label_to_indices: Dict[int, List[int]] = {}
        
        if wildlife_dataset is not None:
            for label, indices in wildlife_dataset.label_to_indices.items():
                self.label_to_indices[label] = indices.copy()
        
        if star_dataset is not None:
            for label, indices in star_dataset.label_to_indices.items():
                # Offset star indices
                offset_indices = [i + self.star_offset for i in indices]
                self.label_to_indices[label] = offset_indices
    
    def __len__(self) -> int:
        return self.wildlife_len + self.star_len
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if idx < self.wildlife_len:
            return self.wildlife_dataset[idx]
        else:
            return self.star_dataset[idx - self.star_offset]
    
    @property
    def num_identities(self) -> int:
        return len(self.label_to_indices)


class CombinedPKSampler(Sampler):
    """
    P-K sampler for combined datasets.
    
    Supports:
    - Mixed batches (samples from both datasets in each batch)
    - Alternating batches (alternates between datasets)
    - Single dataset (wildlife or star only)
    """
    
    def __init__(
        self,
        wildlife_label_to_indices: Optional[Dict[int, List[int]]] = None,
        star_label_to_indices: Optional[Dict[int, List[int]]] = None,
        batch_size: int = 32,
        num_instances: int = 4,
        star_batch_ratio: float = 0.3,
        alternate_batches: bool = False,
    ):
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_identities = batch_size // num_instances
        self.star_batch_ratio = star_batch_ratio
        self.alternate_batches = alternate_batches
        
        # Separate valid labels
        self.wildlife_labels = []
        self.star_labels = []
        
        self.wildlife_label_to_indices = wildlife_label_to_indices or {}
        self.star_label_to_indices = star_label_to_indices or {}
        
        for label, indices in self.wildlife_label_to_indices.items():
            if len(indices) >= num_instances:
                self.wildlife_labels.append(label)
        
        for label, indices in self.star_label_to_indices.items():
            if len(indices) >= num_instances:
                self.star_labels.append(label)
        
        # Calculate batch composition
        if self.alternate_batches:
            self.wildlife_per_batch = self.num_identities
            self.star_per_batch = self.num_identities
        else:
            self.star_per_batch = max(1, int(self.num_identities * star_batch_ratio))
            self.wildlife_per_batch = self.num_identities - self.star_per_batch
        
        # Calculate epoch length
        self._length = self._calculate_length()
        
        print(f"\nCombinedPKSampler:")
        print(f"  Wildlife identities: {len(self.wildlife_labels)}")
        print(f"  Star identities: {len(self.star_labels)}")
        if self.alternate_batches:
            print(f"  Mode: Alternating batches")
        else:
            print(f"  Mode: Mixed batches ({self.wildlife_per_batch} wildlife + {self.star_per_batch} star)")
        print(f"  Batches per epoch: {self._length}")
    
    def _calculate_length(self) -> int:
        """Calculate batches per epoch."""
        if self.alternate_batches:
            wildlife_batches = len(self.wildlife_labels) // self.num_identities
            star_batches = len(self.star_labels) // self.num_identities
            return wildlife_batches + star_batches
        else:
            # Based on smaller dataset to ensure both are sampled
            wildlife_batches = sum(
                len(self.wildlife_label_to_indices[l]) 
                for l in self.wildlife_labels
            ) // (self.wildlife_per_batch * self.num_instances)
            
            star_batches = sum(
                len(self.star_label_to_indices[l])
                for l in self.star_labels
            ) // (self.star_per_batch * self.num_instances) if self.star_labels else float('inf')
            
            return max(1, min(wildlife_batches, star_batches))
    
    def _sample_from_labels(
        self,
        labels: List[int],
        label_to_indices: Dict[int, List[int]],
        num_ids: int,
        pools: Dict[int, List[int]],
        label_idx: int,
    ) -> Tuple[List[int], int]:
        """Sample instances from a set of labels."""
        batch = []
        
        for _ in range(num_ids):
            if not labels:
                break
            
            label = labels[label_idx % len(labels)]
            label_idx += 1
            
            pool = pools.get(label, [])
            if len(pool) < self.num_instances:
                pool = label_to_indices[label].copy()
                random.shuffle(pool)
                pools[label] = pool
            
            indices = [pools[label].pop() for _ in range(self.num_instances)]
            batch.extend(indices)
        
        return batch, label_idx
    
    def __iter__(self):
        """Generate batches."""
        # Shuffle labels
        wildlife_labels = self.wildlife_labels.copy()
        star_labels = self.star_labels.copy()
        random.shuffle(wildlife_labels)
        random.shuffle(star_labels)
        
        # Index pools
        wildlife_pools = {}
        star_pools = {}
        
        wildlife_idx = 0
        star_idx = 0
        
        if self.alternate_batches:
            # Alternate between wildlife and star batches
            for batch_num in range(self._length):
                if batch_num % 2 == 0 and wildlife_labels:
                    batch, wildlife_idx = self._sample_from_labels(
                        wildlife_labels, self.wildlife_label_to_indices,
                        self.num_identities, wildlife_pools, wildlife_idx
                    )
                elif star_labels:
                    batch, star_idx = self._sample_from_labels(
                        star_labels, self.star_label_to_indices,
                        self.num_identities, star_pools, star_idx
                    )
                else:
                    continue
                
                if len(batch) >= self.num_instances * 2:
                    random.shuffle(batch)
                    yield batch
        else:
            # Mixed batches
            for _ in range(self._length):
                batch = []
                
                # Sample from wildlife
                if wildlife_labels and self.wildlife_per_batch > 0:
                    wildlife_batch, wildlife_idx = self._sample_from_labels(
                        wildlife_labels, self.wildlife_label_to_indices,
                        self.wildlife_per_batch, wildlife_pools, wildlife_idx
                    )
                    batch.extend(wildlife_batch)
                
                # Sample from stars
                if star_labels and self.star_per_batch > 0:
                    star_batch, star_idx = self._sample_from_labels(
                        star_labels, self.star_label_to_indices,
                        self.star_per_batch, star_pools, star_idx
                    )
                    batch.extend(star_batch)
                
                if len(batch) >= self.num_instances * 2:
                    random.shuffle(batch)
                    yield batch
    
    def __len__(self) -> int:
        return self._length


def create_pretrain_dataloaders(
    config,
    train_transform=None,
    test_transform=None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create dataloaders for pre-training on Wildlife10k.
    
    Excludes SeaStarReID2023 by default for fair evaluation.
    Uses standard torchvision transforms suitable for diverse species.
    """
    if train_transform is None:
        train_transform = get_wildlife_train_transforms(config.model.image_size)
    if test_transform is None:
        test_transform = get_wildlife_test_transforms(config.model.image_size)
    
    # Build Wildlife10k config
    wildlife_config = Wildlife10kConfig(
        data_root=config.wildlife_root,
        filter=FilterConfig(
            include_datasets=config.include_datasets if config.include_datasets else [],
            exclude_datasets=config.exclude_datasets,
        ),
        split=SplitConfig(strategy=config.split_strategy),
        image_size=config.model.image_size,
        batch_size=config.batch_size,
        num_instances=config.num_instances,
    )
    
    # Load datasets
    print("\nLoading Wildlife10k for pre-training...")
    train_dataset = Wildlife10kDataset(
        data_root=config.wildlife_root,
        transform=train_transform,
        mode='train',
        config=wildlife_config,
    )
    
    test_dataset = Wildlife10kDataset(
        data_root=config.wildlife_root,
        transform=test_transform,
        mode='test',
        config=wildlife_config,
    )
    
    train_dataset.print_summary()
    test_dataset.print_summary()
    
    # Create sampler
    from wildlife_reid.utils.samplers import PKSampler
    train_sampler = PKSampler(
        label_to_indices=train_dataset.label_to_indices,
        batch_size=config.batch_size,
        num_instances=config.num_instances,
    )
    
    # Dataloader optimization settings
    loader_kwargs = {
        'num_workers': config.num_workers,
        'pin_memory': config.pin_memory,
    }
    if config.num_workers > 0:
        loader_kwargs['persistent_workers'] = True
        loader_kwargs['prefetch_factor'] = 4 if not IS_WINDOWS else 2
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        **loader_kwargs,
    )
    
    # Evaluation batch size - architecture-aware multiplier
    # Transformers (SwinV2) need smaller multiplier due to attention memory
    eval_multiplier = get_eval_batch_multiplier(config.model.backbone)
    eval_batch_size = int(config.batch_size * eval_multiplier)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    
    return train_loader, test_loader


def create_finetune_dataloaders(
    config,
    train_transform=None,
    test_transform=None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create dataloaders for fine-tuning on star_dataset.
    
    Returns train, gallery (train split), and query (test split) loaders.
    Uses sophisticated Albumentations transforms optimized for underwater sea stars.
    """
    if train_transform is None:
        train_transform = get_star_train_transforms(config.model.image_size)
    if test_transform is None:
        test_transform = get_star_test_transforms(config.model.image_size)
    
    print("\nLoading star_dataset for fine-tuning...")
    
    # Training dataset
    train_dataset = StarDataset(
        data_root=config.star_dataset_root,
        transform=train_transform,
        mode='train',
        train_outing_ratio=config.train_outing_ratio,
        min_outings_for_eval=config.min_outings_for_eval,
        seed=config.seed,
    )
    train_dataset.print_summary()
    
    # Gallery (train split for comparison)
    gallery_dataset = StarDataset(
        data_root=config.star_dataset_root,
        transform=test_transform,
        mode='train',
        train_outing_ratio=config.train_outing_ratio,
        min_outings_for_eval=config.min_outings_for_eval,
        seed=config.seed,
    )
    
    # Query (test split)
    query_dataset = StarDataset(
        data_root=config.star_dataset_root,
        transform=test_transform,
        mode='test',
        train_outing_ratio=config.train_outing_ratio,
        min_outings_for_eval=config.min_outings_for_eval,
        seed=config.seed,
    )
    query_dataset.print_summary()
    
    # Filter valid labels for sampling
    # Check if we should include negative-only identities
    include_negative_only = getattr(config, 'include_negative_only', False)
    
    if include_negative_only:
        # Include all identities with enough instances (including negative-only)
        valid_labels = [
            l for l, indices in train_dataset.label_to_indices.items()
            if len(indices) >= config.num_instances
        ]
        print(f"  Including negative-only identities in training")
    else:
        # Exclude negative-only identities (default behavior)
        valid_labels = [
            l for l, indices in train_dataset.label_to_indices.items()
            if len(indices) >= config.num_instances and l not in train_dataset.negative_only_labels
        ]
    
    n_negative_only_valid = len([
        l for l, indices in train_dataset.label_to_indices.items()
        if len(indices) >= config.num_instances and l in train_dataset.negative_only_labels
    ])
    print(f"PKSampler: {len(valid_labels)} valid identities" + 
          (f" (+{n_negative_only_valid} negative-only)" if include_negative_only else 
           f" ({n_negative_only_valid} negative-only excluded)") +
          f", {config.num_instances} per batch, {len(valid_labels) * config.num_instances // config.batch_size} batches")
    
    if len(valid_labels) < config.batch_size // config.num_instances:
        raise ValueError(
            f"Not enough valid identities ({len(valid_labels)}) for batch size "
            f"{config.batch_size} with {config.num_instances} instances each"
        )
    
    from wildlife_reid.utils.samplers import PKSampler
    train_sampler = PKSampler(
        label_to_indices={l: train_dataset.label_to_indices[l] for l in valid_labels},
        batch_size=config.batch_size,
        num_instances=config.num_instances,
    )
    
    # Dataloader optimization settings
    loader_kwargs = {
        'num_workers': config.num_workers,
        'pin_memory': config.pin_memory,
    }
    if config.num_workers > 0:
        # persistent_workers keeps workers alive between epochs (huge speedup)
        # prefetch_factor controls how many batches each worker prefetches
        loader_kwargs['persistent_workers'] = True
        loader_kwargs['prefetch_factor'] = 4 if not IS_WINDOWS else 2
    
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        **loader_kwargs,
    )
    
    # Evaluation batch size - architecture-aware multiplier
    # Transformers (SwinV2) need smaller multiplier due to attention memory
    eval_multiplier = get_eval_batch_multiplier(config.model.backbone)
    eval_batch_size = int(config.batch_size * eval_multiplier)
    
    gallery_loader = DataLoader(
        gallery_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    
    query_loader = DataLoader(
        query_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    
    return train_loader, gallery_loader, query_loader


def create_cotrain_dataloaders(
    config,
    train_transform=None,
    test_transform=None,
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    """
    Create dataloaders for co-training on Wildlife10k + star_dataset.
    
    Uses:
    - Wildlife10k: Standard torchvision transforms
    - star_dataset: Sophisticated Albumentations transforms (underwater-optimized)
    
    Returns:
        train_loader: Combined training data
        wildlife_test_loader: Wildlife10k test set
        star_gallery_loader: Star dataset gallery
        star_query_loader: Star dataset query
    """
    # Use appropriate transforms for each dataset type
    wildlife_train_transform = get_wildlife_train_transforms(config.model.image_size)
    wildlife_test_transform = get_wildlife_test_transforms(config.model.image_size)
    star_train_transform = get_star_train_transforms(config.model.image_size)
    star_test_transform = get_star_test_transforms(config.model.image_size)
    
    # Load Wildlife10k
    print("\nLoading Wildlife10k for co-training...")
    wildlife_config = Wildlife10kConfig(
        data_root=config.wildlife_root,
        filter=FilterConfig(
            include_datasets=config.include_datasets if config.include_datasets else [],
            exclude_datasets=config.exclude_datasets,
        ),
        split=SplitConfig(strategy=config.split_strategy),
        image_size=config.model.image_size,
    )
    
    wildlife_train = Wildlife10kDataset(
        data_root=config.wildlife_root,
        transform=wildlife_train_transform,
        mode='train',
        config=wildlife_config,
    )
    wildlife_train.print_summary()
    
    wildlife_test = Wildlife10kDataset(
        data_root=config.wildlife_root,
        transform=wildlife_test_transform,
        mode='test',
        config=wildlife_config,
    )
    
    # Load star_dataset with label offset
    print("\nLoading star_dataset for co-training...")
    wildlife_max_label = max(wildlife_train.label_to_indices.keys()) + 1
    
    star_train = StarDataset(
        data_root=config.star_dataset_root,
        transform=star_train_transform,
        mode='train',
        train_outing_ratio=config.train_outing_ratio,
        min_outings_for_eval=config.min_outings_for_eval,
        seed=config.seed,
        label_offset=wildlife_max_label,
    )
    star_train.print_summary()
    
    star_gallery = StarDataset(
        data_root=config.star_dataset_root,
        transform=star_test_transform,
        mode='train',
        train_outing_ratio=config.train_outing_ratio,
        min_outings_for_eval=config.min_outings_for_eval,
        seed=config.seed,
        label_offset=wildlife_max_label,
    )
    
    star_query = StarDataset(
        data_root=config.star_dataset_root,
        transform=star_test_transform,
        mode='test',
        train_outing_ratio=config.train_outing_ratio,
        min_outings_for_eval=config.min_outings_for_eval,
        seed=config.seed,
        label_offset=wildlife_max_label,
    )
    
    # Combined dataset
    combined_train = CombinedDataset(wildlife_train, star_train, mode='train')
    
    # Check if we should include negative-only identities
    include_negative_only = getattr(config, 'include_negative_only', False)
    
    # Filter star labels
    if include_negative_only:
        # Include all identities with enough instances (including negative-only)
        valid_star_labels = {
            l: star_train.label_to_indices[l]
            for l in star_train.label_to_indices
            if len(star_train.label_to_indices[l]) >= config.num_instances
        }
        print(f"  Including negative-only identities in star training")
    else:
        # Exclude negative-only identities (default behavior)
        valid_star_labels = {
            l: star_train.label_to_indices[l]
            for l in star_train.label_to_indices
            if len(star_train.label_to_indices[l]) >= config.num_instances
            and l not in star_train.negative_only_labels
        }
    
    # Create combined sampler
    train_sampler = CombinedPKSampler(
        wildlife_label_to_indices=wildlife_train.label_to_indices,
        star_label_to_indices=valid_star_labels,
        batch_size=config.batch_size,
        num_instances=config.num_instances,
        star_batch_ratio=config.star_batch_ratio,
        alternate_batches=config.alternate_batches,
    )
    
    # Dataloader optimization settings
    loader_kwargs = {
        'num_workers': config.num_workers,
        'pin_memory': config.pin_memory,
    }
    if config.num_workers > 0:
        loader_kwargs['persistent_workers'] = True
        loader_kwargs['prefetch_factor'] = 4 if not IS_WINDOWS else 2
    
    # Create loaders
    train_loader = DataLoader(
        combined_train,
        batch_sampler=train_sampler,
        **loader_kwargs,
    )
    
    # Evaluation batch size - architecture-aware multiplier
    # Transformers (SwinV2) need smaller multiplier due to attention memory
    eval_multiplier = get_eval_batch_multiplier(config.model.backbone)
    eval_batch_size = int(config.batch_size * eval_multiplier)
    
    wildlife_test_loader = DataLoader(
        wildlife_test,
        batch_size=eval_batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    
    star_gallery_loader = DataLoader(
        star_gallery,
        batch_size=eval_batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    
    star_query_loader = DataLoader(
        star_query,
        batch_size=eval_batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    
    return train_loader, wildlife_test_loader, star_gallery_loader, star_query_loader

