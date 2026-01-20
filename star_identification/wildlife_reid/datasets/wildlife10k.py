"""
Main Wildlife10k Dataset class.
"""
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np

from .base import BaseWildlifeDataset
from .subdataset import SubDatasetHandler
from ..registry import DATASET_REGISTRY
from ..config import Wildlife10kConfig, FilterConfig, SplitConfig


class Wildlife10kDataset(BaseWildlifeDataset):
    """
    PyTorch Dataset for Wildlife ReID-10k.
    
    Handles:
    - Loading from metadata.csv
    - Filtering by dataset/species
    - Per-dataset split strategies
    - Unified identity management across sub-datasets
    """
    
    def __init__(
        self,
        data_root: str,
        transform=None,
        mode: str = 'train',
        config: Optional[Wildlife10kConfig] = None,
        filter_config: Optional[FilterConfig] = None,
        split_config: Optional[SplitConfig] = None,
    ):
        """
        Args:
            data_root: Path to wildlifeReID folder
            transform: Image transforms
            mode: 'train' or 'test'
            config: Full config (overrides filter_config and split_config)
            filter_config: Filtering options
            split_config: Split strategy options
        """
        self.data_root = Path(data_root)
        self.mode = mode
        self.subdataset_handler = SubDatasetHandler()
        
        # Use config values or defaults
        if config is not None:
            filter_config = config.filter
            split_config = config.split
        
        self.filter_config = filter_config or FilterConfig()
        self.split_config = split_config or SplitConfig()
        
        # Load and prepare data
        df = self._load_and_prepare_data()
        
        # Call parent init
        super().__init__(df, transform, mode)
        
        # Build per-dataset statistics
        self._build_dataset_info()
    
    def _load_and_prepare_data(self) -> pd.DataFrame:
        """Load metadata and apply filtering/splitting."""
        
        # Load metadata
        metadata_path = self.data_root / 'metadata.csv'
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found at {metadata_path}")
        
        df = pd.read_csv(metadata_path, low_memory=False)
        print(f"Loaded {len(df)} total images from metadata.csv")
        
        # Convert relative paths to absolute
        df['path'] = df['path'].apply(lambda x: str(self.data_root / x))
        
        # Prefix identities with dataset name to avoid collisions
        df['identity'] = df['dataset'] + '_' + df['identity'].astype(str)
        
        # Apply filters
        df = self._apply_filters(df)
        
        # Apply per-dataset split strategies if not using original
        if self.split_config.strategy != "original":
            df = self._apply_split_strategies(df)
        
        # Filter by minimum images
        df = self._filter_by_min_images(df)
        
        return df
    
    def _apply_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply filtering based on filter config."""
        original_len = len(df)
        
        # Filter by dataset
        if self.filter_config.include_datasets:
            df = df[df['dataset'].isin(self.filter_config.include_datasets)]
        
        if self.filter_config.exclude_datasets:
            df = df[~df['dataset'].isin(self.filter_config.exclude_datasets)]
        
        # Filter by species
        if self.filter_config.include_species:
            df = df[df['species'].isin(self.filter_config.include_species)]
        
        if self.filter_config.exclude_species:
            df = df[~df['species'].isin(self.filter_config.exclude_species)]
        
        # Filter by required fields
        if self.filter_config.require_orientation:
            df = df[df['orientation'].notna()]
        
        if self.filter_config.require_date:
            df = df[df['date'].notna()]
        
        if len(df) < original_len:
            print(f"Filtered: {original_len} -> {len(df)} images")
        
        return df
    
    def _apply_split_strategies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply per-dataset split strategies."""
        print(f"\nApplying split strategy: {self.split_config.strategy}")
        
        result_dfs = []
        
        for dataset_name in df['dataset'].unique():
            dataset_df = df[df['dataset'] == dataset_name].copy()
            
            # Determine strategy for this dataset
            if self.split_config.strategy == "recommended":
                # Use the registry's recommended strategy
                strategy = self.subdataset_handler.get_recommended_strategy(dataset_name)
            else:
                # Use the global strategy
                strategy = self.split_config.strategy
            
            # Apply split
            dataset_df = self.subdataset_handler.apply_split(
                dataset_df,
                dataset_name,
                strategy=strategy,
                train_ratio=self.split_config.train_ratio,
                seed=self.split_config.seed,
            )
            
            result_dfs.append(dataset_df)
        
        return pd.concat(result_dfs, ignore_index=True)
    
    def _filter_by_min_images(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter identities by minimum image count."""
        min_images = self.split_config.min_images_per_identity
        
        # Count images per identity
        id_counts = df['identity'].value_counts()
        valid_ids = id_counts[id_counts >= min_images].index
        
        original_ids = df['identity'].nunique()
        df = df[df['identity'].isin(valid_ids)]
        
        if df['identity'].nunique() < original_ids:
            print(f"Filtered identities by min_images={min_images}: "
                  f"{original_ids} -> {df['identity'].nunique()}")
        
        return df
    
    def _build_dataset_info(self):
        """Build per-dataset statistics."""
        self.dataset_info = {}
        
        for dataset_name in self.df['dataset'].unique():
            ds_df = self.df[self.df['dataset'] == dataset_name]
            
            self.dataset_info[dataset_name] = {
                'images': len(ds_df),
                'identities': ds_df['identity'].nunique(),
                'species': ds_df['species'].iloc[0] if len(ds_df) > 0 else 'unknown',
                'registry_info': DATASET_REGISTRY.get(dataset_name),
            }
    
    def get_dataset_weights(self) -> Dict[str, float]:
        """
        Get sampling weights for balancing across datasets.
        
        Returns inverse frequency weights so smaller datasets
        are sampled more frequently.
        """
        total = len(self.df)
        weights = {}
        
        for ds_name, info in self.dataset_info.items():
            # Inverse frequency
            weights[ds_name] = total / (info['images'] * len(self.dataset_info))
        
        # Normalize
        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}
        
        return weights
    
    def get_species_weights(self) -> Dict[str, float]:
        """Get sampling weights for balancing across species."""
        species_counts = self.df['species'].value_counts()
        total = len(self.df)
        
        weights = {}
        for species, count in species_counts.items():
            weights[species] = total / (count * len(species_counts))
        
        # Normalize
        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}
        
        return weights
    
    def get_sample_weights(self) -> np.ndarray:
        """
        Get per-sample weights for weighted sampling.
        
        Balances by dataset to prevent large datasets from dominating.
        """
        dataset_weights = self.get_dataset_weights()
        
        weights = np.zeros(len(self.df))
        for idx, row in enumerate(self.df.itertuples()):
            weights[idx] = dataset_weights[row.dataset]
        
        return weights
    
    @classmethod
    def from_config(
        cls,
        config: Wildlife10kConfig,
        transform=None,
        mode: str = 'train',
    ) -> 'Wildlife10kDataset':
        """Create dataset from config."""
        return cls(
            data_root=config.data_root,
            transform=transform,
            mode=mode,
            config=config,
        )
    
    def subset_by_dataset(self, dataset_names: List[str]) -> 'Wildlife10kDataset':
        """Create a new dataset containing only specified sub-datasets."""
        new_df = self.df[self.df['dataset'].isin(dataset_names)].copy()
        
        # Create new instance with filtered df
        new_instance = object.__new__(Wildlife10kDataset)
        new_instance.data_root = self.data_root
        new_instance.mode = self.mode
        new_instance.filter_config = self.filter_config
        new_instance.split_config = self.split_config
        new_instance.subdataset_handler = self.subdataset_handler
        
        # Initialize parent attributes
        BaseWildlifeDataset.__init__(
            new_instance, 
            new_df, 
            self.transform, 
            self.mode
        )
        new_instance._build_dataset_info()
        
        return new_instance
    
    def subset_by_species(self, species_names: List[str]) -> 'Wildlife10kDataset':
        """Create a new dataset containing only specified species."""
        new_df = self.df[self.df['species'].isin(species_names)].copy()
        
        # Create new instance with filtered df
        new_instance = object.__new__(Wildlife10kDataset)
        new_instance.data_root = self.data_root
        new_instance.mode = self.mode
        new_instance.filter_config = self.filter_config
        new_instance.split_config = self.split_config
        new_instance.subdataset_handler = self.subdataset_handler
        
        # Initialize parent attributes
        BaseWildlifeDataset.__init__(
            new_instance,
            new_df,
            self.transform,
            self.mode
        )
        new_instance._build_dataset_info()
        
        return new_instance


def create_wildlife10k_dataloaders(
    config: Wildlife10kConfig,
    train_transform,
    test_transform,
) -> Tuple[Any, Any]:
    """
    Create train and test dataloaders for Wildlife10k.
    
    Args:
        config: Wildlife10k config
        train_transform: Training augmentations
        test_transform: Test/val transforms
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    from torch.utils.data import DataLoader
    from ..utils.samplers import PKSampler
    
    # Create datasets
    train_dataset = Wildlife10kDataset.from_config(
        config, transform=train_transform, mode='train'
    )
    test_dataset = Wildlife10kDataset.from_config(
        config, transform=test_transform, mode='test'
    )
    
    train_dataset.print_summary()
    test_dataset.print_summary()
    
    # Create sampler for P-K sampling
    train_sampler = PKSampler(
        label_to_indices=train_dataset.label_to_indices,
        batch_size=config.batch_size,
        num_instances=config.num_instances,
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size * 2,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    
    return train_loader, test_loader


