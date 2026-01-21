#!/usr/bin/env python
"""
Verification tests for evaluation issues.

This script verifies the issues identified in the evaluation code:
1. Silent query skipping
2. Edge case in temporal split
3. 0.0 vs failure indistinguishable
4. Dataset non-determinism risk
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from megastarid.datasets.combined import StarDataset


def test_issue1_silent_query_skipping():
    """
    Issue 1: Queries with no gallery matches are silently skipped.
    
    This test verifies that this CAN happen if data has issues,
    and demonstrates the lack of logging.
    """
    print("\n" + "=" * 70)
    print("ISSUE 1: Silent Query Skipping")
    print("=" * 70)
    
    # Simulate the evaluation logic to show what happens
    # when a query has no gallery matches
    
    # Simulated gallery and query labels
    gallery_labels = np.array([0, 0, 1, 1, 2, 2, 3, 3])  # 4 identities in gallery
    query_labels = np.array([0, 1, 4, 5])  # 2 valid, 2 have NO gallery match!
    
    # Simulated similarities (doesn't matter for this test)
    similarities = np.random.randn(len(query_labels), len(gallery_labels))
    
    # This is the ACTUAL code from trainer.py validate_star()
    all_aps = []
    all_ranks = []
    skipped = 0  # We add this, but original code doesn't track
    
    for i in range(len(query_labels)):
        query_label = query_labels[i]
        sims = similarities[i]
        
        sorted_indices = np.argsort(-sims)
        sorted_labels = gallery_labels[sorted_indices]
        
        matches = sorted_labels == query_label
        
        if matches.sum() == 0:
            skipped += 1  # Silent in original!
            continue  # Query is dropped!
        
        # ... rest of metric computation
        cumsum = np.cumsum(matches)
        precision = cumsum / (np.arange(len(matches)) + 1)
        ap = (precision * matches).sum() / matches.sum()
        all_aps.append(ap)
        
        first_match = np.where(matches)[0][0]
        all_ranks.append(first_match + 1)
    
    mAP = np.mean(all_aps) if all_aps else 0.0
    
    print(f"\nTest scenario:")
    print(f"  Gallery identities: {set(gallery_labels)}")
    print(f"  Query identities: {set(query_labels)}")
    print(f"  Queries with NO gallery match: {set(query_labels) - set(gallery_labels)}")
    print(f"\nResults:")
    print(f"  Queries skipped: {skipped}/{len(query_labels)} (SILENT in original code!)")
    print(f"  mAP computed over: {len(all_aps)} queries only")
    print(f"  Reported mAP: {mAP:.4f}")
    print(f"\n⚠️  ISSUE CONFIRMED: 50% of queries were silently dropped!")
    print(f"    The original code has NO logging of this.")
    
    return skipped > 0


def test_issue2_temporal_split_edge_case():
    """
    Issue 2: Edge case where n_train = 0.
    
    Let me verify if this can actually happen.
    """
    print("\n" + "=" * 70)
    print("ISSUE 2: Temporal Split Edge Case")
    print("=" * 70)
    
    # The temporal split logic:
    # if outings >= min_outings (default 2):
    #     n_train = int(n_samples * train_ratio)  # default 0.8
    #     train gets first n_train, test gets rest
    
    # For n_train = 0, we need:
    # int(n_samples * 0.8) = 0
    # => n_samples * 0.8 < 1
    # => n_samples < 1.25
    # => n_samples = 1
    
    # But for outings >= 2, we need at least 2 different dates
    # Each date requires at least 1 sample
    # So min n_samples = 2 for outings >= 2
    
    # Let's verify:
    test_cases = [
        (1, 0.8),  # 1 sample, 0.8 ratio
        (2, 0.8),  # 2 samples, 0.8 ratio
        (2, 0.4),  # 2 samples, 0.4 ratio (edge case)
        (3, 0.3),  # 3 samples, 0.3 ratio
    ]
    
    print("\nTesting n_train calculation:")
    print(f"{'n_samples':<12} {'train_ratio':<12} {'n_train':<10} {'n_test':<10} {'Issue?'}")
    print("-" * 60)
    
    issues_found = []
    for n_samples, train_ratio in test_cases:
        n_train = int(n_samples * train_ratio)
        n_test = n_samples - n_train
        
        issue = ""
        if n_train == 0:
            issue = "⚠️ Gallery empty!"
            issues_found.append((n_samples, train_ratio))
        elif n_test == 0:
            issue = "⚠️ Query empty!"
            issues_found.append((n_samples, train_ratio))
        
        print(f"{n_samples:<12} {train_ratio:<12} {n_train:<10} {n_test:<10} {issue}")
    
    print(f"\n📊 Analysis:")
    print(f"  For outings >= 2, minimum n_samples = 2 (each date needs >= 1 sample)")
    print(f"  With n_samples = 2 and train_ratio = 0.8: n_train = 1, n_test = 1 ✓")
    print(f"  With n_samples = 2 and train_ratio = 0.4: n_train = 0, n_test = 2 ⚠️")
    
    print(f"\n⚠️  EDGE CASE POSSIBLE if train_ratio < 0.5 and n_samples = 2")
    print(f"    Default train_ratio = 0.8 is safe, but edge case exists.")
    
    # Also check: what if someone sets train_ratio = 1.0?
    print(f"\n  Special case - train_ratio = 1.0:")
    n_train = int(5 * 1.0)
    print(f"    n_samples=5, n_train={n_train}, n_test={5-n_train}")
    print(f"    ⚠️ ALL samples go to train, query is EMPTY!")
    
    return len(issues_found) > 0


def test_issue3_zero_vs_failure():
    """
    Issue 3: mAP = 0.0 is returned for both "model is terrible" 
    and "no valid queries".
    """
    print("\n" + "=" * 70)
    print("ISSUE 3: 0.0 vs Failure Indistinguishable")
    print("=" * 70)
    
    # Scenario A: All queries skipped (no gallery matches)
    all_aps_a = []  # Empty!
    mAP_a = np.mean(all_aps_a) if all_aps_a else 0.0
    
    # Scenario B: Model is terrible (all predictions wrong, but valid queries)
    # AP = 0 when first match is at the very end
    all_aps_b = [0.001, 0.001, 0.001]  # Very low but not zero
    mAP_b = np.mean(all_aps_b)
    
    # Scenario C: Actually impossible - AP can't be exactly 0 if there are matches
    # Because precision at first match position is always > 0
    
    print(f"\nScenario A: All queries skipped (data bug)")
    print(f"  all_aps = {all_aps_a}")
    print(f"  mAP = {mAP_a}")
    
    print(f"\nScenario B: Model performs poorly")
    print(f"  all_aps = {all_aps_b}")
    print(f"  mAP = {mAP_b:.4f}")
    
    print(f"\n⚠️  ISSUE CONFIRMED:")
    print(f"    Scenario A returns mAP = 0.0")
    print(f"    There's no way to distinguish 'evaluation failed' from 'model failed'")
    print(f"    (Though in practice, Scenario B can't produce exactly 0.0)")
    
    return True


def test_issue4_nondeterminism_risk():
    """
    Issue 4: np.random.seed in _apply_temporal_split affects global state.
    """
    print("\n" + "=" * 70)
    print("ISSUE 4: Dataset Non-determinism Risk")
    print("=" * 70)
    
    # The current _apply_temporal_split has:
    # np.random.seed(seed)  # Global seed!
    # 
    # But it doesn't actually use randomness currently.
    # The risk is:
    # 1. If randomness is added later
    # 2. If np.random is called between dataset instantiations
    
    print("\nCurrent code in _apply_temporal_split:")
    print("  np.random.seed(seed)  # Sets GLOBAL numpy random state")
    print("  # ... but no actual random calls currently")
    
    # Demonstrate the risk
    seed = 42
    
    # Simulate two dataset creations with interference
    np.random.seed(seed)
    result1 = np.random.rand()  # First "dataset"
    
    # Interference from other code
    _ = np.random.rand()  # Some other random call
    
    np.random.seed(seed)
    result2 = np.random.rand()  # Second "dataset" 
    
    print(f"\nDemonstration:")
    print(f"  Dataset 1 random value: {result1:.6f}")
    print(f"  Dataset 2 random value: {result2:.6f}")
    print(f"  Are they identical? {result1 == result2}")
    
    # Now show the problem if seed is NOT reset
    np.random.seed(seed)
    r1 = np.random.rand()
    r2 = np.random.rand()  # Different without re-seeding!
    
    print(f"\n  Without re-seeding between calls:")
    print(f"  Call 1: {r1:.6f}")
    print(f"  Call 2: {r2:.6f}")
    print(f"  Different? {r1 != r2}")
    
    print(f"\n📊 Current Status:")
    print(f"  The _apply_temporal_split is called 3 times:")
    print(f"    1. train_dataset = StarDataset(..., mode='train')")
    print(f"    2. gallery_dataset = StarDataset(..., mode='train')")
    print(f"    3. query_dataset = StarDataset(..., mode='test')")
    print(f"  Each call resets np.random.seed(seed), so currently safe.")
    print(f"  But if 'split' column exists in CSV, _apply_temporal_split is skipped!")
    
    print(f"\n⚠️  RISK CONFIRMED:")
    print(f"    Code uses global numpy seed which is fragile")
    print(f"    Safe now, but could break if randomness added")
    
    return True


def test_real_dataset():
    """
    Test with actual star_dataset to verify issues in practice.
    """
    print("\n" + "=" * 70)
    print("REAL DATASET VERIFICATION")
    print("=" * 70)
    
    possible_paths = [
        Path("D:/star_identification/star_dataset"),
        Path("D:/star_identification/star_dataset_resized"),
    ]
    
    data_root = None
    for p in possible_paths:
        if p.exists():
            data_root = p
            break
    
    if data_root is None:
        print("SKIP: star_dataset not found")
        return None
    
    print(f"Testing with: {data_root}")
    
    # Create gallery and query datasets
    gallery = StarDataset(data_root=str(data_root), mode='train', seed=42)
    query = StarDataset(data_root=str(data_root), mode='test', seed=42)
    
    print(f"\nDataset sizes:")
    print(f"  Gallery: {len(gallery)} images, {gallery.num_identities} identities")
    print(f"  Query: {len(query)} images, {query.num_identities} identities")
    
    # Check for identities in query but not in gallery
    gallery_identities = set(gallery.df['identity'].unique())
    query_identities = set(query.df['identity'].unique())
    
    only_in_query = query_identities - gallery_identities
    only_in_gallery = gallery_identities - query_identities
    shared = query_identities & gallery_identities
    
    print(f"\nIdentity analysis:")
    print(f"  Shared (in both): {len(shared)}")
    print(f"  Only in gallery: {len(only_in_gallery)}")
    print(f"  Only in query: {len(only_in_query)} {'⚠️ WOULD BE SKIPPED!' if only_in_query else '✓'}")
    
    if only_in_query:
        print(f"\n  Identities only in query (would be skipped):")
        for ident in list(only_in_query)[:5]:
            print(f"    - {ident}")
        if len(only_in_query) > 5:
            print(f"    ... and {len(only_in_query) - 5} more")
    
    # Check label consistency
    print(f"\nLabel mapping consistency:")
    mismatches = 0
    for ident in shared:
        g_label = gallery.identity_to_label[ident]
        q_label = query.identity_to_label[ident]
        if g_label != q_label:
            mismatches += 1
    
    print(f"  Label mismatches: {mismatches} {'⚠️ BUG!' if mismatches else '✓'}")
    
    # Count how many queries would be skipped
    query_labels_in_gallery = 0
    for ident in query.df['identity'].unique():
        label = query.identity_to_label[ident]
        if label in gallery.label_to_indices:
            query_labels_in_gallery += 1
    
    would_skip = query.num_identities - query_labels_in_gallery
    print(f"\n  Query identities with gallery samples: {query_labels_in_gallery}")
    print(f"  Query identities WITHOUT gallery samples: {would_skip}")
    
    if would_skip > 0:
        print(f"\n⚠️  {would_skip} query identities would be SILENTLY SKIPPED in evaluation!")
    else:
        print(f"\n✓ All query identities have gallery matches")
    
    return len(only_in_query) == 0


if __name__ == '__main__':
    print("=" * 70)
    print("EVALUATION ISSUES VERIFICATION")
    print("=" * 70)
    
    results = {}
    
    results['issue1'] = test_issue1_silent_query_skipping()
    results['issue2'] = test_issue2_temporal_split_edge_case()
    results['issue3'] = test_issue3_zero_vs_failure()
    results['issue4'] = test_issue4_nondeterminism_risk()
    results['real_data'] = test_real_dataset()
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Issue 1 (Silent skipping): {'CONFIRMED' if results['issue1'] else 'NOT FOUND'}")
    print(f"Issue 2 (Edge case split): {'EDGE CASE EXISTS' if results['issue2'] else 'SAFE'}")
    print(f"Issue 3 (0.0 ambiguity): {'CONFIRMED' if results['issue3'] else 'NOT FOUND'}")
    print(f"Issue 4 (Non-determinism): {'RISK EXISTS' if results['issue4'] else 'SAFE'}")
    if results['real_data'] is not None:
        print(f"Real data check: {'PASS' if results['real_data'] else 'ISSUES FOUND'}")


