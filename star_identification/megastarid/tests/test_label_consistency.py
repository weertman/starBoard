#!/usr/bin/env python
"""
Test to verify that gallery and query datasets have consistent label mappings.

This test catches the critical bug where identity_to_label was built per-split,
causing label misalignment between gallery (train) and query (test) datasets.
"""
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from megastarid.datasets.combined import StarDataset


def test_label_consistency():
    """
    Verify that the same identity maps to the same label in both train and test splits.
    
    This is critical for correct evaluation - if labels differ between gallery and
    query, the metrics will be completely wrong.
    """
    # Use the actual star_dataset path if it exists, otherwise skip
    # Note: paths relative to where this test is run from
    possible_paths = [
        Path("D:/star_identification/star_dataset"),
        Path("D:/star_identification/star_dataset_resized"),
        Path("./star_dataset"),
        Path("./star_dataset_resized"),
        Path("../star_dataset"),
        Path("../star_dataset_resized"),
    ]
    
    data_root = None
    for p in possible_paths:
        if p.exists():
            data_root = p
            break
    
    if data_root is None:
        print("SKIP: star_dataset not found at expected paths")
        return True
    
    print(f"Testing with star_dataset at: {data_root}")
    
    # Create gallery (train) and query (test) datasets
    gallery_dataset = StarDataset(
        data_root=str(data_root),
        transform=None,
        mode='train',
        seed=42,
    )
    
    query_dataset = StarDataset(
        data_root=str(data_root),
        transform=None,
        mode='test',
        seed=42,
    )
    
    print(f"\nGallery (train): {gallery_dataset.num_images} images, {gallery_dataset.num_identities} identities")
    print(f"Query (test): {query_dataset.num_images} images, {query_dataset.num_identities} identities")
    print(f"Total identities (global): {gallery_dataset.num_identities_global}")
    
    # Check that identity_to_label mappings are identical
    gallery_mapping = gallery_dataset.identity_to_label
    query_mapping = query_dataset.identity_to_label
    
    # Both should have the same keys (all identities)
    if set(gallery_mapping.keys()) != set(query_mapping.keys()):
        print("\n❌ FAIL: identity_to_label has different keys!")
        print(f"  Gallery keys: {len(gallery_mapping)}")
        print(f"  Query keys: {len(query_mapping)}")
        return False
    
    # Every identity should map to the same label
    mismatches = []
    for identity in gallery_mapping:
        if gallery_mapping[identity] != query_mapping[identity]:
            mismatches.append((identity, gallery_mapping[identity], query_mapping[identity]))
    
    if mismatches:
        print(f"\n❌ FAIL: {len(mismatches)} identities have different labels!")
        for identity, g_label, q_label in mismatches[:5]:
            print(f"  {identity}: gallery={g_label}, query={q_label}")
        if len(mismatches) > 5:
            print(f"  ... and {len(mismatches) - 5} more")
        return False
    
    # Verify that shared identities (in both splits) have correct labels
    query_identities = set(query_dataset.df['identity'].unique())
    gallery_identities = set(gallery_dataset.df['identity'].unique())
    shared_identities = query_identities & gallery_identities
    
    print(f"\nShared identities (in both train and test): {len(shared_identities)}")
    
    if len(shared_identities) == 0:
        print("\n⚠️ WARNING: No shared identities between train and test!")
        print("This might indicate a problem with the split.")
    
    # Sample some identities and verify labels match
    for identity in list(shared_identities)[:5]:
        g_label = gallery_mapping[identity]
        q_label = query_mapping[identity]
        print(f"  {identity}: label={g_label} (consistent: {g_label == q_label})")
        assert g_label == q_label, f"Label mismatch for {identity}!"
    
    print("\n✅ PASS: All identity-to-label mappings are consistent!")
    return True


def test_label_offset_consistency():
    """Test that label_offset is applied consistently."""
    possible_paths = [
        Path("D:/star_identification/star_dataset"),
        Path("D:/star_identification/star_dataset_resized"),
        Path("./star_dataset"),
        Path("./star_dataset_resized"),
        Path("../star_dataset"),
        Path("../star_dataset_resized"),
    ]
    
    data_root = None
    for p in possible_paths:
        if p.exists():
            data_root = p
            break
    
    if data_root is None:
        print("SKIP: star_dataset not found")
        return True
    
    print(f"\nTesting label_offset consistency...")
    
    # Create datasets with offset (like in co-training)
    offset = 1000
    
    gallery_dataset = StarDataset(
        data_root=str(data_root),
        transform=None,
        mode='train',
        seed=42,
        label_offset=offset,
    )
    
    query_dataset = StarDataset(
        data_root=str(data_root),
        transform=None,
        mode='test',
        seed=42,
        label_offset=offset,
    )
    
    # All labels should be >= offset
    min_gallery_label = min(gallery_dataset.label_to_indices.keys())
    min_query_label = min(query_dataset.label_to_indices.keys()) if query_dataset.label_to_indices else offset
    
    if min_gallery_label < offset:
        print(f"❌ FAIL: Gallery has labels < offset ({min_gallery_label} < {offset})")
        return False
    
    if min_query_label < offset:
        print(f"❌ FAIL: Query has labels < offset ({min_query_label} < {offset})")
        return False
    
    # Mappings should still be consistent
    for identity in gallery_dataset.identity_to_label:
        g_label = gallery_dataset.identity_to_label[identity]
        q_label = query_dataset.identity_to_label[identity]
        if g_label != q_label:
            print(f"❌ FAIL: Label mismatch with offset for {identity}")
            return False
    
    print(f"✅ PASS: Label offset ({offset}) applied consistently!")
    return True


if __name__ == '__main__':
    success = True
    
    print("=" * 60)
    print("Testing label consistency between gallery and query datasets")
    print("=" * 60)
    
    if not test_label_consistency():
        success = False
    
    if not test_label_offset_consistency():
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("All tests passed!")
    else:
        print("Some tests failed!")
        sys.exit(1)

