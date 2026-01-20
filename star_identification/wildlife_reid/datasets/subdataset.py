"""
Sub-dataset handler for Wildlife10k.

Each sub-dataset may require specific logic for:
- Train/test splitting
- Data cleaning
- Special handling of metadata
"""
from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path
import numpy as np
import pandas as pd

from ..registry import DATASET_REGISTRY, DatasetInfo, SplitStrategy


class SubDatasetHandler:
    """
    Handles sub-dataset specific logic for Wildlife10k.
    
    Provides methods to:
    - Apply dataset-appropriate split strategies
    - Filter problematic samples
    - Validate data quality
    """
    
    def __init__(self, registry=None):
        """
        Args:
            registry: DatasetRegistry instance (uses global if None)
        """
        self.registry = registry or DATASET_REGISTRY
    
    def get_info(self, dataset_name: str) -> Optional[DatasetInfo]:
        """Get dataset info from registry."""
        return self.registry.get(dataset_name)
    
    def apply_split(
        self,
        df: pd.DataFrame,
        dataset_name: str,
        strategy: Optional[str] = None,
        train_ratio: float = 0.8,
        seed: int = 42,
    ) -> pd.DataFrame:
        """
        Apply appropriate split strategy to a sub-dataset.
        
        Args:
            df: DataFrame for a single sub-dataset
            dataset_name: Name of the sub-dataset
            strategy: Override strategy (None = use recommended)
            train_ratio: Train split ratio for random/cluster splits
            seed: Random seed
            
        Returns:
            DataFrame with updated 'split' column
        """
        info = self.get_info(dataset_name)
        
        # Determine strategy
        if strategy is None:
            if info is not None:
                strategy = info.recommended_split.value
            else:
                strategy = "original"
        
        # Apply strategy
        if strategy == "original":
            # Use existing split - no changes needed
            return df
        
        elif strategy == "time_aware":
            return self._apply_time_aware_split(df, train_ratio, seed)
        
        elif strategy == "cluster_aware":
            return self._apply_cluster_aware_split(df, train_ratio, seed)
        
        elif strategy == "random":
            return self._apply_random_split(df, train_ratio, seed)
        
        else:
            raise ValueError(f"Unknown split strategy: {strategy}")
    
    def _apply_time_aware_split(
        self,
        df: pd.DataFrame,
        train_ratio: float,
        seed: int,
    ) -> pd.DataFrame:
        """
        Split based on temporal information.
        
        For each identity, earlier dates go to train, later to test.
        """
        df = df.copy()
        np.random.seed(seed)
        
        if 'date' not in df.columns or df['date'].isna().all():
            # Fallback to random if no dates
            print(f"  Warning: No dates available, falling back to random split")
            return self._apply_random_split(df, train_ratio, seed)
        
        # Parse dates
        df['_parsed_date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Split each identity by time
        for identity in df['identity'].unique():
            mask = df['identity'] == identity
            id_df = df[mask]
            
            # Check if this identity has valid dates
            valid_dates = id_df['_parsed_date'].notna()
            
            if valid_dates.sum() >= 2:
                # Sort by date
                sorted_indices = id_df.loc[valid_dates].sort_values('_parsed_date').index
                n_train = int(len(sorted_indices) * train_ratio)
                
                train_indices = sorted_indices[:n_train]
                test_indices = sorted_indices[n_train:]
                
                df.loc[train_indices, 'split'] = 'train'
                df.loc[test_indices, 'split'] = 'test'
                
                # Handle samples without dates - random assignment
                no_date_mask = mask & ~valid_dates
                if no_date_mask.any():
                    no_date_indices = df[no_date_mask].index
                    n_train_no_date = int(len(no_date_indices) * train_ratio)
                    shuffled = np.random.permutation(no_date_indices)
                    df.loc[shuffled[:n_train_no_date], 'split'] = 'train'
                    df.loc[shuffled[n_train_no_date:], 'split'] = 'test'
            else:
                # Not enough dated samples, use random for this identity
                id_indices = id_df.index.tolist()
                n_train = max(1, int(len(id_indices) * train_ratio))
                shuffled = np.random.permutation(id_indices)
                df.loc[shuffled[:n_train], 'split'] = 'train'
                df.loc[shuffled[n_train:], 'split'] = 'test'
        
        # Cleanup
        df = df.drop(columns=['_parsed_date'])
        
        return df
    
    def _apply_cluster_aware_split(
        self,
        df: pd.DataFrame,
        train_ratio: float,
        seed: int,
    ) -> pd.DataFrame:
        """
        Split based on cluster information.
        
        Clusters of similar images stay together to prevent data leakage.
        """
        df = df.copy()
        np.random.seed(seed)
        
        if 'cluster_id' not in df.columns or df['cluster_id'].isna().all():
            # Fallback to random if no clusters
            print(f"  Warning: No clusters available, falling back to random split")
            return self._apply_random_split(df, train_ratio, seed)
        
        # Split each identity by clusters
        for identity in df['identity'].unique():
            mask = df['identity'] == identity
            id_df = df[mask]
            
            # Get unique clusters for this identity
            has_cluster = id_df['cluster_id'].notna()
            
            if has_cluster.sum() >= 2:
                clusters = id_df.loc[has_cluster, 'cluster_id'].unique()
                
                if len(clusters) >= 2:
                    # Randomly assign clusters to train/test
                    n_train_clusters = max(1, int(len(clusters) * train_ratio))
                    shuffled_clusters = np.random.permutation(clusters)
                    train_clusters = set(shuffled_clusters[:n_train_clusters])
                    
                    # Assign based on cluster
                    for idx in id_df[has_cluster].index:
                        cluster = df.loc[idx, 'cluster_id']
                        df.loc[idx, 'split'] = 'train' if cluster in train_clusters else 'test'
                else:
                    # Only one cluster, split randomly within it
                    clustered_indices = id_df[has_cluster].index.tolist()
                    n_train = max(1, int(len(clustered_indices) * train_ratio))
                    shuffled = np.random.permutation(clustered_indices)
                    df.loc[shuffled[:n_train], 'split'] = 'train'
                    df.loc[shuffled[n_train:], 'split'] = 'test'
                
                # Handle samples without clusters
                no_cluster_mask = mask & ~has_cluster
                if no_cluster_mask.any():
                    no_cluster_indices = df[no_cluster_mask].index.tolist()
                    n_train = max(1, int(len(no_cluster_indices) * train_ratio))
                    shuffled = np.random.permutation(no_cluster_indices)
                    df.loc[shuffled[:n_train], 'split'] = 'train'
                    df.loc[shuffled[n_train:], 'split'] = 'test'
            else:
                # Not enough clustered samples, use random
                id_indices = id_df.index.tolist()
                n_train = max(1, int(len(id_indices) * train_ratio))
                shuffled = np.random.permutation(id_indices)
                df.loc[shuffled[:n_train], 'split'] = 'train'
                df.loc[shuffled[n_train:], 'split'] = 'test'
        
        return df
    
    def _apply_random_split(
        self,
        df: pd.DataFrame,
        train_ratio: float,
        seed: int,
    ) -> pd.DataFrame:
        """
        Apply random per-identity split.
        
        Each identity's samples are randomly split.
        """
        df = df.copy()
        np.random.seed(seed)
        
        for identity in df['identity'].unique():
            mask = df['identity'] == identity
            indices = df[mask].index.tolist()
            
            n_train = max(1, int(len(indices) * train_ratio))
            shuffled = np.random.permutation(indices)
            
            df.loc[shuffled[:n_train], 'split'] = 'train'
            df.loc[shuffled[n_train:], 'split'] = 'test'
        
        return df
    
    def validate_dataset(
        self,
        df: pd.DataFrame,
        dataset_name: str,
        check_paths: bool = False,
    ) -> Dict[str, Any]:
        """
        Validate a sub-dataset and report issues.
        
        Args:
            df: DataFrame for a single sub-dataset
            dataset_name: Name of the sub-dataset
            check_paths: Whether to verify image paths exist
            
        Returns:
            Dictionary with validation results
        """
        results = {
            'dataset': dataset_name,
            'total_images': len(df),
            'issues': [],
            'warnings': [],
        }
        
        # Check for required columns
        required = ['identity', 'path', 'split']
        for col in required:
            if col not in df.columns:
                results['issues'].append(f"Missing required column: {col}")
        
        if results['issues']:
            return results
        
        # Check split distribution
        split_counts = df['split'].value_counts()
        results['train_count'] = split_counts.get('train', 0)
        results['test_count'] = split_counts.get('test', 0)
        
        if results['train_count'] == 0:
            results['issues'].append("No training samples")
        if results['test_count'] == 0:
            results['warnings'].append("No test samples")
        
        # Check identity distribution
        id_counts = df['identity'].value_counts()
        single_image_ids = (id_counts == 1).sum()
        if single_image_ids > 0:
            results['warnings'].append(
                f"{single_image_ids} identities have only 1 image"
            )
        
        # Check for identities appearing in both splits
        train_ids = set(df[df['split'] == 'train']['identity'].unique())
        test_ids = set(df[df['split'] == 'test']['identity'].unique())
        shared_ids = train_ids & test_ids
        
        results['train_only_ids'] = len(train_ids - test_ids)
        results['test_only_ids'] = len(test_ids - train_ids)
        results['shared_ids'] = len(shared_ids)
        
        if len(test_ids - train_ids) > 0:
            results['warnings'].append(
                f"{len(test_ids - train_ids)} identities appear ONLY in test (open-set)"
            )
        
        # Optionally check paths
        if check_paths:
            missing = 0
            for path in df['path'].unique()[:100]:  # Check first 100
                if not Path(path).exists():
                    missing += 1
            if missing > 0:
                results['issues'].append(f"At least {missing} image paths don't exist")
        
        return results
    
    def get_recommended_strategy(self, dataset_name: str) -> str:
        """Get the recommended split strategy for a dataset."""
        info = self.get_info(dataset_name)
        if info is not None:
            return info.recommended_split.value
        return "original"
    
    def filter_valid_identities(
        self,
        df: pd.DataFrame,
        min_images: int = 2,
        require_in_both_splits: bool = False,
    ) -> pd.DataFrame:
        """
        Filter to keep only valid identities.
        
        Args:
            df: DataFrame to filter
            min_images: Minimum images per identity
            require_in_both_splits: If True, identity must be in train AND test
            
        Returns:
            Filtered DataFrame
        """
        df = df.copy()
        
        # Filter by minimum images
        id_counts = df['identity'].value_counts()
        valid_ids = id_counts[id_counts >= min_images].index
        df = df[df['identity'].isin(valid_ids)]
        
        # Filter by split presence
        if require_in_both_splits:
            train_ids = set(df[df['split'] == 'train']['identity'].unique())
            test_ids = set(df[df['split'] == 'test']['identity'].unique())
            shared_ids = train_ids & test_ids
            df = df[df['identity'].isin(shared_ids)]
        
        return df.reset_index(drop=True)


