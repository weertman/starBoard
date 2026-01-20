"""
High-level data loader for Wildlife10k.

Provides a simple interface for loading the dataset with proper configuration.
"""
from typing import Optional, Tuple, Dict, List, Any
from pathlib import Path

import pandas as pd

from .config import Wildlife10kConfig
from .datasets import Wildlife10kDataset
from .utils.transforms import get_train_transforms, get_test_transforms
from .utils.samplers import PKSampler


class Wildlife10kLoader:
    """
    High-level loader for Wildlife10k dataset.
    
    Example usage:
        loader = Wildlife10kLoader("./wildlifeReID")
        train_dataset, test_dataset = loader.load()
        train_loader, test_loader = loader.create_dataloaders()
    """
    
    def __init__(
        self,
        data_root: str,
        config: Optional[Wildlife10kConfig] = None,
    ):
        """
        Args:
            data_root: Path to wildlifeReID folder
            config: Configuration (uses defaults if None)
        """
        self.data_root = Path(data_root)
        self.config = config or Wildlife10kConfig(data_root=str(data_root))
        
        # Datasets (lazy loaded)
        self._train_dataset: Optional[Wildlife10kDataset] = None
        self._test_dataset: Optional[Wildlife10kDataset] = None
        
        # Validate path
        metadata_path = self.data_root / 'metadata.csv'
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"metadata.csv not found at {metadata_path}. "
                f"Please ensure you have the Wildlife10k dataset."
            )
    
    def load(
        self,
        train_transform=None,
        test_transform=None,
    ) -> Tuple[Wildlife10kDataset, Wildlife10kDataset]:
        """
        Load train and test datasets.
        
        Args:
            train_transform: Training transforms (uses default if None)
            test_transform: Test transforms (uses default if None)
            
        Returns:
            Tuple of (train_dataset, test_dataset)
        """
        if train_transform is None:
            train_transform = get_train_transforms(self.config.image_size)
        if test_transform is None:
            test_transform = get_test_transforms(self.config.image_size)
        
        print(f"\nLoading Wildlife10k from {self.data_root}")
        print(f"Config: {self.config.summary()}")
        
        self._train_dataset = Wildlife10kDataset(
            data_root=str(self.data_root),
            transform=train_transform,
            mode='train',
            config=self.config,
        )
        
        self._test_dataset = Wildlife10kDataset(
            data_root=str(self.data_root),
            transform=test_transform,
            mode='test',
            config=self.config,
        )
        
        return self._train_dataset, self._test_dataset
    
    def create_dataloaders(
        self,
        train_transform=None,
        test_transform=None,
    ) -> Tuple[Any, Any]:
        """
        Create PyTorch DataLoaders for training.
        
        Args:
            train_transform: Training transforms
            test_transform: Test transforms
            
        Returns:
            Tuple of (train_loader, test_loader)
        """
        from torch.utils.data import DataLoader
        
        # Load datasets if not already loaded
        if self._train_dataset is None or self._test_dataset is None:
            self.load(train_transform, test_transform)
        
        # Create P-K sampler for training
        train_sampler = PKSampler(
            label_to_indices=self._train_dataset.label_to_indices,
            batch_size=self.config.batch_size,
            num_instances=self.config.num_instances,
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            self._train_dataset,
            batch_sampler=train_sampler,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
        )
        
        test_loader = DataLoader(
            self._test_dataset,
            batch_size=self.config.batch_size * 2,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
        )
        
        return train_loader, test_loader
    
    def get_dataset_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the loaded datasets."""
        if self._train_dataset is None:
            self.load()
        
        return {
            'train': self._train_dataset.get_statistics(),
            'test': self._test_dataset.get_statistics(),
        }
    
    def list_datasets(self) -> List[str]:
        """List available sub-datasets."""
        if self._train_dataset is None:
            self.load()
        
        return list(self._train_dataset.dataset_info.keys())
    
    def list_species(self) -> List[str]:
        """List available species."""
        if self._train_dataset is None:
            self.load()
        
        return sorted(self._train_dataset.df['species'].unique())
    
    def subset(
        self,
        datasets: Optional[List[str]] = None,
        species: Optional[List[str]] = None,
    ) -> Tuple[Wildlife10kDataset, Wildlife10kDataset]:
        """
        Get a subset of the data.
        
        Args:
            datasets: List of dataset names to include
            species: List of species to include
            
        Returns:
            Tuple of (train_subset, test_subset)
        """
        if self._train_dataset is None:
            self.load()
        
        train_subset = self._train_dataset
        test_subset = self._test_dataset
        
        if datasets is not None:
            train_subset = train_subset.subset_by_dataset(datasets)
            test_subset = test_subset.subset_by_dataset(datasets)
        
        if species is not None:
            train_subset = train_subset.subset_by_species(species)
            test_subset = test_subset.subset_by_species(species)
        
        return train_subset, test_subset


def load_wildlife10k(
    data_root: str,
    **config_kwargs,
) -> Tuple[Wildlife10kDataset, Wildlife10kDataset]:
    """
    Convenience function to load Wildlife10k datasets.
    
    Args:
        data_root: Path to wildlifeReID folder
        **config_kwargs: Additional config parameters
        
    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    config = Wildlife10kConfig(data_root=data_root, **config_kwargs)
    loader = Wildlife10kLoader(data_root, config)
    return loader.load()


