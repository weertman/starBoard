#!/usr/bin/env python
"""
Test script for wildlife_reid module.

Run from project root:
    python -m wildlife_reid.test_module
"""
import sys
from pathlib import Path


def test_registry():
    """Test dataset registry."""
    print("\n" + "=" * 60)
    print("TEST: Dataset Registry")
    print("=" * 60)
    
    from wildlife_reid import DATASET_REGISTRY
    
    # Test basic registry functions
    all_datasets = DATASET_REGISTRY.get_all()
    print(f"✓ Registered {len(all_datasets)} datasets")
    
    # Test species lookup
    species = DATASET_REGISTRY.list_species()
    print(f"✓ Found {len(species)} species: {species[:5]}...")
    
    # Test specific dataset lookup
    sea_star = DATASET_REGISTRY.get("SeaStarReID2023")
    assert sea_star is not None
    print(f"✓ SeaStarReID2023: species={sea_star.species}, split={sea_star.recommended_split}")
    
    # Test time-aware datasets
    time_aware = DATASET_REGISTRY.get_time_aware_datasets()
    print(f"✓ Time-aware datasets: {time_aware}")
    
    # Test cluster-aware datasets
    cluster_aware = DATASET_REGISTRY.get_cluster_aware_datasets()
    print(f"✓ Cluster-aware datasets: {len(cluster_aware)} datasets")
    
    return True


def test_config():
    """Test configuration classes."""
    print("\n" + "=" * 60)
    print("TEST: Configuration")
    print("=" * 60)
    
    from wildlife_reid import Wildlife10kConfig, FilterConfig, SplitConfig
    import tempfile
    
    # Create config with custom settings
    config = Wildlife10kConfig(
        data_root="./wildlifeReID",
        image_size=384,
        batch_size=32,
        filter=FilterConfig(include_species=["sea star"]),
        split=SplitConfig(strategy="original"),
    )
    
    print(f"✓ Created config: {config.summary()}")
    
    # Test save/load
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        config.save(f.name)
        loaded = Wildlife10kConfig.load(f.name)
        assert loaded.image_size == config.image_size
        print(f"✓ Config save/load works")
    
    return True


def test_subdataset_handler():
    """Test sub-dataset handling."""
    print("\n" + "=" * 60)
    print("TEST: SubDataset Handler")
    print("=" * 60)
    
    from wildlife_reid.datasets import SubDatasetHandler
    import pandas as pd
    import numpy as np
    
    handler = SubDatasetHandler()
    
    # Create mock dataframe
    np.random.seed(42)
    mock_data = []
    for i in range(100):
        mock_data.append({
            'identity': f'id_{i % 10}',
            'path': f'/fake/path/{i}.jpg',
            'split': 'train' if i < 70 else 'test',
            'date': f'2024-{(i % 12) + 1:02d}-01' if i % 3 == 0 else None,
            'cluster_id': f'cluster_{i % 5}' if i % 2 == 0 else None,
        })
    
    df = pd.DataFrame(mock_data)
    
    # Test random split
    df_random = handler._apply_random_split(df.copy(), train_ratio=0.8, seed=42)
    print(f"✓ Random split: {(df_random['split']=='train').sum()}/{(df_random['split']=='test').sum()}")
    
    # Test time-aware split
    df_time = handler._apply_time_aware_split(df.copy(), train_ratio=0.7, seed=42)
    print(f"✓ Time-aware split: {(df_time['split']=='train').sum()}/{(df_time['split']=='test').sum()}")
    
    # Test cluster-aware split
    df_cluster = handler._apply_cluster_aware_split(df.copy(), train_ratio=0.8, seed=42)
    print(f"✓ Cluster-aware split: {(df_cluster['split']=='train').sum()}/{(df_cluster['split']=='test').sum()}")
    
    # Test validation
    results = handler.validate_dataset(df, "MockDataset")
    print(f"✓ Validation: {len(results['issues'])} issues, {len(results['warnings'])} warnings")
    
    return True


def test_dataset_loading(data_root: str = "./wildlifeReID"):
    """Test actual dataset loading."""
    print("\n" + "=" * 60)
    print("TEST: Dataset Loading")
    print("=" * 60)
    
    from wildlife_reid import Wildlife10kLoader, Wildlife10kConfig, FilterConfig
    
    # Check if data exists
    if not Path(data_root).exists():
        print(f"⚠ Skipping: {data_root} not found")
        return True
    
    # Test with small subset
    config = Wildlife10kConfig(
        data_root=data_root,
        filter=FilterConfig(include_datasets=["SeaStarReID2023"]),
    )
    
    loader = Wildlife10kLoader(data_root, config)
    train_ds, test_ds = loader.load()
    
    print(f"✓ Loaded {len(train_ds)} train, {len(test_ds)} test images")
    print(f"✓ {train_ds.num_identities} train identities, {test_ds.num_identities} test identities")
    
    # Test getting a sample
    sample = train_ds[0]
    assert 'image' in sample
    assert 'label' in sample
    assert 'identity' in sample
    print(f"✓ Sample loaded: {sample['identity']}, label={sample['label']}")
    
    # Test subsetting
    all_datasets = loader.list_datasets()
    all_species = loader.list_species()
    print(f"✓ Found {len(all_datasets)} datasets, {len(all_species)} species")
    
    return True


def test_samplers():
    """Test sampling utilities."""
    print("\n" + "=" * 60)
    print("TEST: Samplers")
    print("=" * 60)
    
    from wildlife_reid.utils.samplers import PKSampler
    
    # Create mock label_to_indices
    label_to_indices = {
        0: list(range(0, 20)),
        1: list(range(20, 40)),
        2: list(range(40, 60)),
        3: list(range(60, 80)),
        4: list(range(80, 100)),
    }
    
    sampler = PKSampler(
        label_to_indices=label_to_indices,
        batch_size=16,
        num_instances=4,
    )
    
    print(f"✓ Created PKSampler: {len(sampler)} batches")
    
    # Test iteration
    batches = list(sampler)
    assert len(batches) > 0
    assert all(len(b) == 16 for b in batches)
    print(f"✓ Generated {len(batches)} batches of size 16")
    
    return True


def test_transforms():
    """Test transform utilities."""
    print("\n" + "=" * 60)
    print("TEST: Transforms")
    print("=" * 60)
    
    from wildlife_reid.utils.transforms import (
        get_train_transforms,
        get_test_transforms,
        get_tta_transforms,
    )
    from PIL import Image
    import torch
    
    # Create test image
    img = Image.new('RGB', (512, 512), color='red')
    
    # Test train transforms
    train_tf = get_train_transforms(image_size=384)
    train_out = train_tf(img)
    assert train_out.shape == (3, 384, 384)
    print(f"✓ Train transform: {train_out.shape}")
    
    # Test test transforms
    test_tf = get_test_transforms(image_size=384)
    test_out = test_tf(img)
    assert test_out.shape == (3, 384, 384)
    print(f"✓ Test transform: {test_out.shape}")
    
    # Test TTA transforms
    tta_tf, n_views = get_tta_transforms(image_size=384)
    tta_out = tta_tf(img)
    assert tta_out.shape == (n_views, 3, 384, 384)
    print(f"✓ TTA transform: {tta_out.shape} ({n_views} views)")
    
    return True


def main():
    """Run all tests."""
    print("Wildlife ReID Module Tests")
    print("=" * 60)
    
    tests = [
        ("Registry", test_registry),
        ("Config", test_config),
        ("SubDataset Handler", test_subdataset_handler),
        ("Samplers", test_samplers),
        ("Transforms", test_transforms),
        ("Dataset Loading", test_dataset_loading),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, success, None))
        except Exception as e:
            results.append((name, False, str(e)))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for name, success, error in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {name}")
        if error:
            print(f"       Error: {error}")
    
    passed = sum(1 for _, s, _ in results if s)
    print(f"\n{passed}/{len(results)} tests passed")
    
    return all(s for _, s, _ in results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


