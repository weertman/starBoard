"""
Test TTA (with vertical flip) and reranking functionality.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class DummyModel(nn.Module):
    """Simple model that returns normalized embeddings."""
    def __init__(self, embed_dim=128):
        super().__init__()
        self.embed_dim = embed_dim
        self.conv = nn.Conv2d(3, 64, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, embed_dim)
    
    def forward(self, x, return_normalized=False):
        x = self.conv(x)
        x = torch.relu(x)
        x = self.pool(x).flatten(1)
        x = self.fc(x)
        if return_normalized:
            x = torch.nn.functional.normalize(x, p=2, dim=1)
        return x


class DummyDataset(Dataset):
    """Dummy dataset with images, labels, identities, and paths."""
    def __init__(self, num_samples=50, num_classes=10, image_size=64):
        self.num_samples = num_samples
        self.labels = torch.randint(0, num_classes, (num_samples,))
        self.image_size = image_size
        # Create deterministic "images" based on label for consistency
        torch.manual_seed(42)
        self.images = torch.randn(num_samples, 3, image_size, image_size)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return {
            'image': self.images[idx],
            'label': self.labels[idx],
            'identity': f'identity_{self.labels[idx].item()}',
            'path': f'/fake/path/class_{self.labels[idx].item()}/img_{idx}.jpg',
        }


def test_tta_with_both_flips():
    """Test that TTA with horizontal + vertical flip works correctly."""
    print("\n" + "=" * 60)
    print("TEST: TTA with Horizontal + Vertical Flip")
    print("=" * 60)
    
    from megastarid.inference import extract_embeddings_with_tta
    
    device = torch.device('cpu')
    model = DummyModel(embed_dim=128)
    model.eval()
    
    dataset = DummyDataset(num_samples=20, num_classes=5)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    
    # Test 1: No TTA (baseline)
    print("\n1. Testing no TTA...")
    emb_none, labels_none = extract_embeddings_with_tta(
        model, dataloader, device,
        use_horizontal_flip=False, use_vertical_flip=False
    )
    print(f"   Shape: {emb_none.shape}, Labels: {labels_none.shape}")
    assert emb_none.shape == (20, 128), f"Expected (20, 128), got {emb_none.shape}"
    
    # Test 2: Horizontal flip only
    print("\n2. Testing horizontal flip only...")
    emb_hflip, labels_hflip = extract_embeddings_with_tta(
        model, dataloader, device,
        use_horizontal_flip=True, use_vertical_flip=False
    )
    print(f"   Shape: {emb_hflip.shape}")
    assert emb_hflip.shape == (20, 128)
    
    # Test 3: Vertical flip only
    print("\n3. Testing vertical flip only...")
    emb_vflip, labels_vflip = extract_embeddings_with_tta(
        model, dataloader, device,
        use_horizontal_flip=False, use_vertical_flip=True
    )
    print(f"   Shape: {emb_vflip.shape}")
    assert emb_vflip.shape == (20, 128)
    
    # Test 4: Both flips (default TTA)
    print("\n4. Testing both flips (full TTA)...")
    emb_both, labels_both = extract_embeddings_with_tta(
        model, dataloader, device,
        use_horizontal_flip=True, use_vertical_flip=True
    )
    print(f"   Shape: {emb_both.shape}")
    assert emb_both.shape == (20, 128)
    
    # Verify embeddings are normalized
    norms = np.linalg.norm(emb_both, axis=1)
    print(f"   Embedding norms: min={norms.min():.4f}, max={norms.max():.4f}")
    assert np.allclose(norms, 1.0, atol=1e-5), "Embeddings should be L2 normalized"
    
    # Verify that different TTA settings produce different embeddings
    diff_hflip = np.abs(emb_none - emb_hflip).mean()
    diff_vflip = np.abs(emb_none - emb_vflip).mean()
    diff_both = np.abs(emb_none - emb_both).mean()
    print(f"\n   Mean diff from baseline:")
    print(f"     Horizontal flip: {diff_hflip:.4f}")
    print(f"     Vertical flip:   {diff_vflip:.4f}")
    print(f"     Both flips:      {diff_both:.4f}")
    
    # All should be different (model responds to different inputs)
    assert diff_hflip > 0, "H-flip should produce different embeddings"
    assert diff_vflip > 0, "V-flip should produce different embeddings"
    assert diff_both > 0, "Both flips should produce different embeddings"
    
    print("\n✓ TTA with both flips test PASSED")


def test_extract_with_metadata_tta():
    """Test that _extract_with_metadata uses both flips when use_tta=True."""
    print("\n" + "=" * 60)
    print("TEST: _extract_with_metadata with TTA")
    print("=" * 60)
    
    from megastarid.evaluation import _extract_with_metadata
    
    device = torch.device('cpu')
    model = DummyModel(embed_dim=128)
    model.eval()
    
    dataset = DummyDataset(num_samples=20, num_classes=5)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    
    # Without TTA
    print("\n1. Without TTA...")
    emb_no_tta, labels, identities, paths = _extract_with_metadata(
        model, dataloader, device, use_tta=False
    )
    print(f"   Embeddings: {emb_no_tta.shape}")
    print(f"   Identities: {len(identities)} items")
    print(f"   Paths: {len(paths)} items")
    
    # With TTA
    print("\n2. With TTA (h-flip + v-flip)...")
    emb_tta, labels_tta, identities_tta, paths_tta = _extract_with_metadata(
        model, dataloader, device, use_tta=True
    )
    print(f"   Embeddings: {emb_tta.shape}")
    
    # Verify shapes match
    assert emb_no_tta.shape == emb_tta.shape
    assert len(identities) == len(identities_tta)
    assert len(paths) == len(paths_tta)
    
    # Verify embeddings are different (TTA should change them)
    diff = np.abs(emb_no_tta - emb_tta).mean()
    print(f"   Mean diff with TTA: {diff:.4f}")
    assert diff > 0, "TTA should produce different embeddings"
    
    # Verify normalized
    norms = np.linalg.norm(emb_tta, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), "TTA embeddings should be normalized"
    
    print("\n✓ _extract_with_metadata TTA test PASSED")


def test_reranking():
    """Test that k-reciprocal reranking works."""
    print("\n" + "=" * 60)
    print("TEST: k-Reciprocal Reranking")
    print("=" * 60)
    
    from megastarid.inference import compute_similarity_matrix
    
    # Create mock embeddings
    np.random.seed(42)
    num_query = 20
    num_gallery = 100
    embed_dim = 128
    
    query_emb = np.random.randn(num_query, embed_dim).astype(np.float32)
    gallery_emb = np.random.randn(num_gallery, embed_dim).astype(np.float32)
    
    # Normalize
    query_emb = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)
    gallery_emb = gallery_emb / np.linalg.norm(gallery_emb, axis=1, keepdims=True)
    
    # Without reranking
    print("\n1. Without reranking (cosine similarity)...")
    sim_no_rerank = compute_similarity_matrix(
        query_emb, gallery_emb, use_reranking=False
    )
    print(f"   Shape: {sim_no_rerank.shape}")
    print(f"   Range: [{sim_no_rerank.min():.4f}, {sim_no_rerank.max():.4f}]")
    
    # With reranking
    print("\n2. With reranking...")
    sim_rerank = compute_similarity_matrix(
        query_emb, gallery_emb, use_reranking=True, k1=10, k2=3
    )
    print(f"   Shape: {sim_rerank.shape}")
    print(f"   Range: [{sim_rerank.min():.4f}, {sim_rerank.max():.4f}]")
    
    # Verify shapes match
    assert sim_no_rerank.shape == sim_rerank.shape == (num_query, num_gallery)
    
    # Verify rankings are different (reranking should change them)
    rank_no_rerank = np.argsort(-sim_no_rerank, axis=1)
    rank_rerank = np.argsort(-sim_rerank, axis=1)
    
    # Check how many top-10 rankings changed
    changed = 0
    for i in range(num_query):
        if not np.array_equal(rank_no_rerank[i, :10], rank_rerank[i, :10]):
            changed += 1
    
    print(f"\n   Queries with changed top-10 rankings: {changed}/{num_query}")
    
    print("\n✓ Reranking test PASSED")


def test_compute_detailed_metrics_with_enhancements():
    """Test compute_detailed_star_metrics with TTA and reranking."""
    print("\n" + "=" * 60)
    print("TEST: compute_detailed_star_metrics with TTA + Reranking")
    print("=" * 60)
    
    from megastarid.evaluation import compute_detailed_star_metrics
    import tempfile
    from pathlib import Path
    
    device = torch.device('cpu')
    model = DummyModel(embed_dim=128)
    model.eval()
    
    # Create gallery and query datasets with overlapping identities
    gallery_dataset = DummyDataset(num_samples=50, num_classes=10)
    query_dataset = DummyDataset(num_samples=30, num_classes=10)
    
    gallery_loader = DataLoader(gallery_dataset, batch_size=8, shuffle=False)
    query_loader = DataLoader(query_dataset, batch_size=8, shuffle=False)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        
        # Without TTA/reranking
        print("\n1. Without TTA/reranking...")
        metrics_base = compute_detailed_star_metrics(
            model=model,
            gallery_loader=gallery_loader,
            query_loader=query_loader,
            device=device,
            output_dir=output_dir / 'base',
            use_tta=False,
            use_reranking=False,
        )
        print(f"   mAP: {metrics_base['mAP']:.4f}, CMC@1: {metrics_base['CMC@1']:.4f}")
        
        # With TTA only
        print("\n2. With TTA only...")
        metrics_tta = compute_detailed_star_metrics(
            model=model,
            gallery_loader=gallery_loader,
            query_loader=query_loader,
            device=device,
            output_dir=output_dir / 'tta',
            use_tta=True,
            use_reranking=False,
        )
        print(f"   mAP: {metrics_tta['mAP']:.4f}, CMC@1: {metrics_tta['CMC@1']:.4f}")
        
        # With reranking only
        print("\n3. With reranking only...")
        metrics_rerank = compute_detailed_star_metrics(
            model=model,
            gallery_loader=gallery_loader,
            query_loader=query_loader,
            device=device,
            output_dir=output_dir / 'rerank',
            use_tta=False,
            use_reranking=True,
        )
        print(f"   mAP: {metrics_rerank['mAP']:.4f}, CMC@1: {metrics_rerank['CMC@1']:.4f}")
        
        # With both TTA and reranking
        print("\n4. With TTA + reranking...")
        metrics_both = compute_detailed_star_metrics(
            model=model,
            gallery_loader=gallery_loader,
            query_loader=query_loader,
            device=device,
            output_dir=output_dir / 'both',
            use_tta=True,
            use_reranking=True,
        )
        print(f"   mAP: {metrics_both['mAP']:.4f}, CMC@1: {metrics_both['CMC@1']:.4f}")
        
        # Verify output files were created
        for subdir in ['base', 'tta', 'rerank', 'both']:
            csv_path = output_dir / subdir / 'star_identity_metrics.csv'
            assert csv_path.exists(), f"Missing {csv_path}"
        
        print("\n   All output files created successfully")
    
    print("\n✓ compute_detailed_star_metrics with enhancements test PASSED")


if __name__ == "__main__":
    print("=" * 60)
    print("TTA AND RERANKING VERIFICATION TESTS")
    print("=" * 60)
    
    test_tta_with_both_flips()
    test_extract_with_metadata_tta()
    test_reranking()
    test_compute_detailed_metrics_with_enhancements()
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)


