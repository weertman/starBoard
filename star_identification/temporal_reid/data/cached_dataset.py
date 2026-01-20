"""
RAM-cached dataset for fast evaluation.

With sufficient RAM (e.g., 1TB), we can pre-load all evaluation images
into memory, eliminating I/O bottleneck during embedding extraction.
"""
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Callable
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class CachedReIDDataset(Dataset):
    """
    Dataset that caches all images in RAM for fast evaluation.
    
    Images are loaded and transformed once, then stored as tensors.
    Subsequent accesses are pure memory reads - extremely fast.
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        transform: Callable,
        identity_to_label: Dict[str, int],
        num_workers: int = 8,
        show_progress: bool = True,
    ):
        """
        Args:
            df: DataFrame with columns [identity, path, ...]
            transform: Image transform (applied during caching)
            identity_to_label: Mapping from identity string to numeric label
            num_workers: Threads for parallel image loading
            show_progress: Show progress bar during caching
        """
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.identity_to_label = identity_to_label
        self.num_workers = num_workers
        
        # Pre-allocate storage
        self.cached_images: List[Optional[torch.Tensor]] = [None] * len(self.df)
        self.labels: List[int] = []
        self.identities: List[str] = []
        self.is_cached = False
        
        # Pre-compute labels and identities (fast)
        for idx, row in self.df.iterrows():
            self.labels.append(identity_to_label[row['identity']])
            self.identities.append(row['identity'])
    
    def cache_all(self, show_progress: bool = True) -> float:
        """
        Load and cache all images in parallel.
        
        Returns:
            Time taken in seconds
        """
        start_time = time.time()
        
        def load_and_transform(idx: int) -> tuple:
            row = self.df.iloc[idx]
            try:
                image = Image.open(row['path']).convert('RGB')
                tensor = self.transform(image)
                return idx, tensor
            except Exception as e:
                print(f"Error loading {row['path']}: {e}")
                # Return dummy tensor
                return idx, torch.zeros(3, 384, 384)
        
        # Parallel loading
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {executor.submit(load_and_transform, i): i for i in range(len(self.df))}
            
            iterator = as_completed(futures)
            if show_progress:
                iterator = tqdm(iterator, total=len(futures), desc="Caching images", leave=False)
            
            for future in iterator:
                idx, tensor = future.result()
                self.cached_images[idx] = tensor
        
        self.is_cached = True
        elapsed = time.time() - start_time
        
        # Calculate memory usage
        sample_tensor = self.cached_images[0]
        bytes_per_image = sample_tensor.element_size() * sample_tensor.numel()
        total_gb = (bytes_per_image * len(self.df)) / (1024**3)
        
        print(f"Cached {len(self.df)} images in {elapsed:.1f}s ({total_gb:.2f} GB RAM)")
        return elapsed
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict:
        if not self.is_cached:
            raise RuntimeError("Dataset not cached! Call cache_all() first.")
        
        return {
            'image': self.cached_images[idx],
            'label': self.labels[idx],
            'identity': self.identities[idx],
            'negative_only': False,  # Eval datasets don't use this
        }


class BackgroundCacher:
    """
    Caches evaluation datasets in background while training runs.
    
    Usage:
        cacher = BackgroundCacher(query_ds, gallery_ds, num_workers=16)
        cacher.start()  # Start background caching
        
        # ... training epoch runs ...
        
        cacher.wait()  # Block until caching complete
        # Now evaluation is instant!
    """
    
    def __init__(
        self,
        *datasets: CachedReIDDataset,
        num_workers: int = 8,
    ):
        self.datasets = datasets
        self.num_workers = num_workers
        self.thread: Optional[threading.Thread] = None
        self.is_complete = threading.Event()
        self.elapsed_time = 0.0
    
    def _cache_worker(self):
        """Background thread that caches all datasets."""
        total_start = time.time()
        
        for ds in self.datasets:
            if not ds.is_cached:
                ds.cache_all(show_progress=True)
        
        self.elapsed_time = time.time() - total_start
        self.is_complete.set()
    
    def start(self):
        """Start background caching."""
        if self.thread is not None and self.thread.is_alive():
            return  # Already running
        
        self.is_complete.clear()
        self.thread = threading.Thread(target=self._cache_worker, daemon=True)
        self.thread.start()
        print("Background caching started...")
    
    def wait(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for caching to complete.
        
        Args:
            timeout: Max seconds to wait (None = wait forever)
        
        Returns:
            True if complete, False if timed out
        """
        if self.is_complete.is_set():
            return True
        
        print("Waiting for background caching to complete...")
        result = self.is_complete.wait(timeout=timeout)
        
        if result:
            print(f"Background caching complete ({self.elapsed_time:.1f}s total)")
        
        return result
    
    @property
    def ready(self) -> bool:
        """Check if all datasets are cached."""
        return self.is_complete.is_set()


def create_cached_eval_loaders(
    df: pd.DataFrame,
    val_transform: Callable,
    identity_to_label: Dict[str, int],
    batch_size: int = 128,
    num_workers: int = 16,
    pin_memory: bool = True,
) -> tuple:
    """
    Create RAM-cached evaluation dataloaders.
    
    Args:
        df: Full metadata DataFrame
        val_transform: Validation transform
        identity_to_label: Label mapping
        batch_size: Eval batch size (can be large, no gradients!)
        num_workers: Workers for initial caching
        pin_memory: Pin memory for faster GPU transfer
    
    Returns:
        (gallery_loader, query_loader, cacher)
        
        Call cacher.start() to begin background caching,
        then cacher.wait() before evaluation.
    """
    # Filter for evaluable identities only
    evaluable_df = df[df['negative_only'] == False]
    
    # Split into gallery (train) and query (test)
    gallery_df = evaluable_df[evaluable_df['split'] == 'train']
    query_df = evaluable_df[evaluable_df['split'] == 'test']
    
    # Create cached datasets
    gallery_dataset = CachedReIDDataset(
        gallery_df, val_transform, identity_to_label, num_workers
    )
    query_dataset = CachedReIDDataset(
        query_df, val_transform, identity_to_label, num_workers
    )
    
    # Create background cacher
    cacher = BackgroundCacher(gallery_dataset, query_dataset, num_workers=num_workers)
    
    # Create dataloaders (no workers needed - data is in RAM!)
    # Using num_workers=0 since we're reading from RAM tensors
    gallery_loader = DataLoader(
        gallery_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # No I/O workers needed
        pin_memory=pin_memory,
    )
    
    query_loader = DataLoader(
        query_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )
    
    return gallery_loader, query_loader, cacher



