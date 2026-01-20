"""
Samplers for Wildlife ReID training.
"""
import random
from typing import Dict, List, Iterator, Optional
import numpy as np

from torch.utils.data import Sampler


class PKSampler(Sampler):
    """
    P-K sampler for metric learning.
    
    Samples P identities per batch, K instances per identity.
    This ensures each batch contains multiple samples per identity,
    which is required for triplet/circle loss.
    """
    
    def __init__(
        self,
        label_to_indices: Dict[int, List[int]],
        batch_size: int,
        num_instances: int = 4,
        drop_last: bool = True,
    ):
        """
        Args:
            label_to_indices: Mapping from label (int) to list of sample indices
            batch_size: Total batch size (must be divisible by num_instances)
            num_instances: Number of instances per identity (K)
            drop_last: Whether to drop incomplete batches
        """
        self.label_to_indices = label_to_indices
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.drop_last = drop_last
        
        # Number of identities per batch (P)
        self.num_identities = batch_size // num_instances
        
        if batch_size % num_instances != 0:
            raise ValueError(
                f"batch_size ({batch_size}) must be divisible by "
                f"num_instances ({num_instances})"
            )
        
        # Filter labels with enough samples
        self.valid_labels = [
            label for label, indices in label_to_indices.items()
            if len(indices) >= num_instances
        ]
        
        if len(self.valid_labels) < self.num_identities:
            raise ValueError(
                f"Not enough identities with >= {num_instances} samples. "
                f"Found {len(self.valid_labels)}, need {self.num_identities}. "
                f"Try reducing num_instances or batch_size."
            )
        
        # Calculate length
        self._length = self._calculate_length()
        
        print(f"PKSampler: {len(self.valid_labels)} valid identities, "
              f"{self.num_identities} per batch, {self._length} batches")
    
    def _calculate_length(self) -> int:
        """Calculate number of batches per epoch."""
        # Total available instances (sampling with replacement)
        total_instances = sum(
            len(self.label_to_indices[label])
            for label in self.valid_labels
        )
        
        # Approximate number of batches to see each sample once
        return max(1, total_instances // self.batch_size)
    
    def __iter__(self) -> Iterator[List[int]]:
        """Generate batches."""
        # Shuffle labels each epoch
        labels = self.valid_labels.copy()
        random.shuffle(labels)
        
        # Create label index pool (for cycling through labels)
        label_pool = labels.copy()
        
        batch_count = 0
        while batch_count < self._length:
            batch = []
            selected_labels = []
            
            # Select P identities
            while len(selected_labels) < self.num_identities:
                if not label_pool:
                    label_pool = labels.copy()
                    random.shuffle(label_pool)
                
                label = label_pool.pop()
                selected_labels.append(label)
            
            # For each identity, sample K instances
            for label in selected_labels:
                indices = self.label_to_indices[label]
                
                if len(indices) >= self.num_instances:
                    # Sample without replacement if possible
                    sampled = random.sample(indices, self.num_instances)
                else:
                    # Sample with replacement if not enough
                    sampled = random.choices(indices, k=self.num_instances)
                
                batch.extend(sampled)
            
            # Verify batch has multiple labels
            if len(set(selected_labels)) >= 2:
                yield batch
                batch_count += 1
    
    def __len__(self) -> int:
        return self._length


class BalancedDatasetSampler(Sampler):
    """
    Sampler that balances across multiple datasets.
    
    Useful when some datasets are much larger than others
    to prevent the large ones from dominating training.
    """
    
    def __init__(
        self,
        dataset_indices: Dict[str, List[int]],
        samples_per_dataset: Optional[int] = None,
        shuffle: bool = True,
    ):
        """
        Args:
            dataset_indices: Mapping from dataset name to list of indices
            samples_per_dataset: Fixed samples per dataset (None = use min)
            shuffle: Whether to shuffle samples
        """
        self.dataset_indices = dataset_indices
        self.shuffle = shuffle
        
        # Determine samples per dataset
        if samples_per_dataset is None:
            # Use minimum dataset size
            samples_per_dataset = min(len(v) for v in dataset_indices.values())
        
        self.samples_per_dataset = samples_per_dataset
        self.num_datasets = len(dataset_indices)
        
        print(f"BalancedDatasetSampler: {self.num_datasets} datasets, "
              f"{samples_per_dataset} samples each")
    
    def __iter__(self) -> Iterator[int]:
        """Generate balanced indices."""
        all_indices = []
        
        for dataset_name, indices in self.dataset_indices.items():
            if self.shuffle:
                sampled = random.sample(
                    indices,
                    min(len(indices), self.samples_per_dataset)
                )
            else:
                sampled = indices[:self.samples_per_dataset]
            
            all_indices.extend(sampled)
        
        if self.shuffle:
            random.shuffle(all_indices)
        
        return iter(all_indices)
    
    def __len__(self) -> int:
        return self.num_datasets * self.samples_per_dataset


