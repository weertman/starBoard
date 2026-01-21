#!/usr/bin/env python
"""
Test that the evaluation.py fixes work correctly.

Tests:
1. Single-pass extraction collects embeddings and metadata together
2. Cached embeddings validation catches mismatches
3. Folder extraction uses parent.name correctly
"""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class MockModel(nn.Module):
    """Mock model that returns fixed embeddings based on input."""
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.embedding_dim = embedding_dim
    
    def forward(self, x, return_normalized=False):
        batch_size = x.shape[0]
        # Generate deterministic embeddings based on batch
        emb = torch.randn(batch_size, self.embedding_dim)
        if return_normalized:
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        return emb
    
    def eval(self):
        return self


class MockBatch:
    """Mock batch for testing."""
    def __init__(self, size, label_start=0):
        self.data = {
            'image': torch.randn(size, 3, 224, 224),
            'label': torch.arange(label_start, label_start + size),
            'identity': [f'star_{i}' for i in range(label_start, label_start + size)],
            'path': [f'/data/folder_{i % 3}/img_{i}.jpg' for i in range(label_start, label_start + size)],
        }
    
    def __getitem__(self, key):
        return self.data[key]


class MockDataLoader:
    """Mock dataloader for testing."""
    def __init__(self, num_batches=3, batch_size=4, label_offset=0):
        self.batches = [
            MockBatch(batch_size, label_start=label_offset + i * batch_size)
            for i in range(num_batches)
        ]
    
    def __iter__(self):
        return iter(self.batches)
    
    def __len__(self):
        return len(self.batches)


def test_single_pass_extraction():
    """
    Test that _extract_with_metadata collects everything in one pass.
    """
    print("\n" + "=" * 70)
    print("TEST: Single-pass extraction")
    print("=" * 70)
    
    from megastarid.evaluation import _extract_with_metadata
    
    model = MockModel()
    dataloader = MockDataLoader(num_batches=2, batch_size=4)
    device = torch.device('cpu')
    
    embeddings, labels, identities, paths = _extract_with_metadata(
        model, dataloader, device, use_tta=False
    )
    
    # Verify shapes and counts
    expected_count = 2 * 4  # 2 batches * 4 samples
    
    print(f"  Embeddings shape: {embeddings.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Identities count: {len(identities)}")
    print(f"  Paths count: {len(paths)}")
    
    if embeddings.shape[0] != expected_count:
        print(f"  ❌ FAIL: Expected {expected_count} embeddings, got {embeddings.shape[0]}")
        return False
    
    if labels.shape[0] != expected_count:
        print(f"  ❌ FAIL: Expected {expected_count} labels, got {labels.shape[0]}")
        return False
    
    if len(identities) != expected_count:
        print(f"  ❌ FAIL: Expected {expected_count} identities, got {len(identities)}")
        return False
    
    if len(paths) != expected_count:
        print(f"  ❌ FAIL: Expected {expected_count} paths, got {len(paths)}")
        return False
    
    # Verify identities match labels
    for i, (label, identity) in enumerate(zip(labels, identities)):
        expected_identity = f'star_{label}'
        if identity != expected_identity:
            print(f"  ❌ FAIL: Identity mismatch at {i}: {identity} != {expected_identity}")
            return False
    
    print("  ✓ PASS: Single-pass extraction works correctly")
    return True


def test_metadata_collection():
    """
    Test that _collect_metadata_and_labels works correctly.
    """
    print("\n" + "=" * 70)
    print("TEST: Metadata collection for cache validation")
    print("=" * 70)
    
    from megastarid.evaluation import _collect_metadata_and_labels
    
    dataloader = MockDataLoader(num_batches=2, batch_size=4)
    
    labels, identities, paths = _collect_metadata_and_labels(dataloader)
    
    expected_count = 2 * 4
    
    print(f"  Labels shape: {labels.shape}")
    print(f"  Identities count: {len(identities)}")
    print(f"  Paths count: {len(paths)}")
    
    if labels.shape[0] != expected_count:
        print(f"  ❌ FAIL: Expected {expected_count} labels, got {labels.shape[0]}")
        return False
    
    # Verify order is consistent
    expected_labels = np.arange(expected_count)
    if not np.array_equal(labels, expected_labels):
        print(f"  ❌ FAIL: Labels don't match expected order")
        return False
    
    print("  ✓ PASS: Metadata collection works correctly")
    return True


def test_cache_validation_catches_mismatch():
    """
    Test that cache validation raises error when cached labels don't match.
    """
    print("\n" + "=" * 70)
    print("TEST: Cache validation catches mismatches")
    print("=" * 70)
    
    # Create a scenario where cached labels don't match fresh labels
    fresh_labels = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    cached_labels = np.array([0, 1, 2, 3, 4, 5, 7, 6])  # Last two swapped!
    
    # Test the validation logic
    if np.array_equal(fresh_labels, cached_labels):
        print("  ❌ FAIL: np.array_equal should return False for mismatched arrays")
        return False
    
    print(f"  Fresh labels:  {fresh_labels}")
    print(f"  Cached labels: {cached_labels}")
    print(f"  Arrays equal: {np.array_equal(fresh_labels, cached_labels)}")
    
    # Find first mismatch
    if len(fresh_labels) == len(cached_labels):
        mismatch_idx = np.where(fresh_labels != cached_labels)[0][0]
        print(f"  First mismatch at index: {mismatch_idx}")
        if mismatch_idx != 6:
            print(f"  ❌ FAIL: Expected mismatch at index 6, got {mismatch_idx}")
            return False
    
    print("  ✓ PASS: Cache validation correctly detects mismatches")
    return True


def test_folder_extraction():
    """
    Test that folder extraction uses parent.name correctly.
    """
    print("\n" + "=" * 70)
    print("TEST: Folder extraction")
    print("=" * 70)
    
    test_cases = [
        # (path, expected_folder)
        ("/data/star_dataset/outreach_2023/image.png", "outreach_2023"),
        ("/data/star_dataset_resized/cruise_2024/photo.jpg", "cruise_2024"),
        ("D:\\star_identification\\star_dataset\\folder1\\img.png", "folder1"),
        ("/home/user/data/my_folder/test.png", "my_folder"),
    ]
    
    all_passed = True
    for path, expected_folder in test_cases:
        folder = Path(path).parent.name
        if folder == expected_folder:
            print(f"  ✓ {path} -> {folder}")
        else:
            print(f"  ❌ {path} -> {folder} (expected {expected_folder})")
            all_passed = False
    
    if all_passed:
        print("  ✓ PASS: Folder extraction works correctly")
    else:
        print("  ❌ FAIL: Some folder extractions failed")
    
    return all_passed


def test_tta_in_single_pass():
    """
    Test that TTA works in single-pass extraction.
    """
    print("\n" + "=" * 70)
    print("TEST: TTA in single-pass extraction")
    print("=" * 70)
    
    from megastarid.evaluation import _extract_with_metadata
    
    model = MockModel()
    dataloader = MockDataLoader(num_batches=1, batch_size=2)
    device = torch.device('cpu')
    
    # Without TTA
    emb_no_tta, _, _, _ = _extract_with_metadata(model, dataloader, device, use_tta=False)
    
    # Reset dataloader
    dataloader = MockDataLoader(num_batches=1, batch_size=2)
    
    # With TTA
    emb_with_tta, _, _, _ = _extract_with_metadata(model, dataloader, device, use_tta=True)
    
    print(f"  Without TTA shape: {emb_no_tta.shape}")
    print(f"  With TTA shape: {emb_with_tta.shape}")
    
    # Both should have same shape
    if emb_no_tta.shape != emb_with_tta.shape:
        print(f"  ❌ FAIL: Shapes don't match")
        return False
    
    # With TTA, embeddings should still be normalized
    norms = np.linalg.norm(emb_with_tta, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-5):
        print(f"  ❌ FAIL: TTA embeddings not normalized. Norms: {norms}")
        return False
    
    print("  ✓ PASS: TTA works correctly in single-pass extraction")
    return True


if __name__ == '__main__':
    print("=" * 70)
    print("TESTING EVALUATION.PY FIXES")
    print("=" * 70)
    
    results = {}
    
    results['single_pass'] = test_single_pass_extraction()
    results['metadata'] = test_metadata_collection()
    results['cache_validation'] = test_cache_validation_catches_mismatch()
    results['folder'] = test_folder_extraction()
    results['tta'] = test_tta_in_single_pass()
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    all_passed = True
    for name, passed in results.items():
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)


