"""
Temporal Sensitivity Analysis for MegaStarID.

Analyzes model sensitivity across explicit time encounters (outings),
tracking how embedding distances vary with image perturbations AND time gaps
between query and gallery observations.

Key Features:
- Explicit outing-to-outing pairing with dates
- Time gap calculation (days between encounters)
- Aggregation by identity AND by time gap buckets
- Per-outing-pair results for fine-grained analysis

Usage:
    python -m megastarid_alignment_utils.temporal_sensitivity_analysis \
        --checkpoint checkpoints/megastarid/best.pth \
        --data-root star_dataset_resized \
        --output temporal_sensitivity_results \
        --num-identities 30 \
        --backbone densenet121
"""

import argparse
import json
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from megastarid.config import FinetuneConfig
from megastarid.models import load_pretrained_model


# Perturbation configurations (reduced for efficiency)
PERTURBATION_CONFIGS = {
    'rotation': {
        'name': 'Rotation',
        'params': np.array([-90, -45, -30, 0, 30, 45, 90]),
        'unit': '°',
    },
    'scale': {
        'name': 'Scale',
        'params': np.array([0.6, 0.8, 1.0, 1.2, 1.4]),
        'unit': '×',
    },
    'brightness': {
        'name': 'Brightness',
        'params': np.array([-0.4, -0.2, 0, 0.2, 0.4]),
        'unit': '',
    },
    'contrast': {
        'name': 'Contrast',
        'params': np.array([0.6, 0.8, 1.0, 1.2, 1.4]),
        'unit': '×',
    },
    'saturation': {
        'name': 'Saturation',
        'params': np.array([0.5, 0.75, 1.0, 1.25, 1.5]),
        'unit': '×',
    },
    'hue': {
        'name': 'Hue Shift',
        'params': np.array([-0.3, -0.15, 0, 0.15, 0.3]),
        'unit': '',
    },
    'gaussian_blur': {
        'name': 'Gaussian Blur',
        'params': np.array([0, 2, 4, 6, 8]),
        'unit': 'σ',
    },
    'jpeg_quality': {
        'name': 'JPEG Quality',
        'params': np.array([100, 70, 50, 30, 10]),
        'unit': '%',
    },
    'noise': {
        'name': 'Gaussian Noise',
        'params': np.array([0, 0.03, 0.06, 0.09, 0.12]),
        'unit': 'σ',
    },
}

# Time gap buckets for aggregation
TIME_GAP_BUCKETS = [
    (0, 30, '0-30 days'),
    (30, 90, '30-90 days'),
    (90, 180, '90-180 days'),
    (180, 365, '180-365 days'),
    (365, 730, '1-2 years'),
    (730, float('inf'), '2+ years'),
]


@dataclass
class OutingInfo:
    """Information about a single outing (time encounter)."""
    outing_id: str
    date: datetime
    image_paths: List[str]
    split: str  # 'train' or 'test'


@dataclass
class OutingPairResult:
    """Sensitivity results for a single query-gallery outing pair."""
    identity: str
    
    # Query outing (test)
    query_outing_id: str
    query_date: datetime
    query_path: str
    
    # Gallery outing (train)
    gallery_outing_id: str
    gallery_date: datetime
    gallery_path: str
    
    # Time gap
    time_gap_days: int
    time_gap_bucket: str
    
    # Results
    baseline_distance: float
    perturbation_results: Dict[str, Dict[str, np.ndarray]]  # {pert: {params, distances}}
    sensitivity_ranges: Dict[str, float]  # {pert: range}


@dataclass
class IdentityTemporalResults:
    """All outing pair results for a single identity."""
    identity: str
    outings: Dict[str, OutingInfo]  # outing_id -> OutingInfo
    outing_pairs: List[OutingPairResult]
    
    # Aggregated across pairs
    mean_baseline: float
    std_baseline: float
    mean_sensitivity: Dict[str, float]
    std_sensitivity: Dict[str, float]


@dataclass
class TemporalAnalysisResults:
    """Complete temporal sensitivity analysis results."""
    num_identities: int
    num_outing_pairs: int
    checkpoint_path: str
    data_root: str
    
    # Per-identity results
    identity_results: List[IdentityTemporalResults]
    
    # All outing pairs (flattened for time gap analysis)
    all_outing_pairs: List[OutingPairResult]
    
    # Aggregated by time gap
    time_gap_aggregations: Dict[str, Dict[str, Any]]
    
    # Overall aggregation
    overall_mean_baseline: float
    overall_std_baseline: float
    overall_sensitivity: Dict[str, Dict[str, float]]  # {pert: {mean, std}}


class TemporalSensitivityAnalyzer:
    """
    Sensitivity analyzer with explicit temporal (outing) tracking.
    
    Analyzes how model sensitivity varies:
    1. Across different perturbations
    2. Across different time gaps between observations
    3. Across different identities
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        data_root: str,
        backbone: str = 'densenet121',
        device: str = 'cuda',
        image_size: int = 384,
        seed: int = 42,
    ):
        self.checkpoint_path = checkpoint_path
        self.data_root = Path(data_root)
        self.backbone = backbone
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.image_size = image_size
        self.seed = seed
        
        # Load model
        print(f"Loading model from {checkpoint_path}...")
        self.model = self._load_model()
        self.model.eval()
        print(f"Model loaded on {self.device}")
        
        # Load metadata with temporal info
        self.metadata = self._load_metadata()
        
        # Find identities with multiple outings across train/test
        self.identity_outings = self._build_identity_outings()
        print(f"Found {len(self.identity_outings)} identities with cross-outing pairs")
        
        # Pre-build transforms
        self._build_base_transform()
    
    def _load_model(self):
        """Load model from checkpoint."""
        config = FinetuneConfig()
        config.model.backbone = self.backbone
        config.model.image_size = self.image_size
        
        model = load_pretrained_model(
            config=config,
            checkpoint_path=self.checkpoint_path,
            device=self.device,
            strict=False,
        )
        return model
    
    def _load_metadata(self) -> pd.DataFrame:
        """Load dataset metadata with temporal info."""
        # Prefer temporal metadata
        metadata_path = self.data_root / 'metadata_temporal.csv'
        if not metadata_path.exists():
            metadata_path = self.data_root / 'metadata.csv'
        
        df = pd.read_csv(metadata_path)
        
        # Parse dates
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        return df
    
    def _build_identity_outings(self) -> Dict[str, Dict[str, OutingInfo]]:
        """
        Build mapping of identity -> outings with temporal info.
        
        Returns:
            {identity: {outing_id: OutingInfo}}
        """
        identity_outings = {}
        
        for identity, group in self.metadata.groupby('identity'):
            outings = {}
            
            for outing_id, outing_group in group.groupby('outing'):
                # Get date (use first non-null)
                dates = outing_group['date'].dropna()
                if len(dates) > 0:
                    date = dates.iloc[0]
                    if isinstance(date, str):
                        date = pd.to_datetime(date)
                else:
                    # Skip outings without dates
                    continue
                
                # Get split
                splits = outing_group['split'].unique()
                split = splits[0] if len(splits) == 1 else 'mixed'
                
                # Get valid paths
                paths = outing_group['path'].tolist()
                paths = [p for p in paths if Path(p).exists()]
                
                if paths:
                    outings[outing_id] = OutingInfo(
                        outing_id=outing_id,
                        date=date,
                        image_paths=paths,
                        split=split,
                    )
            
            # Only keep identities with both train and test outings
            train_outings = [o for o in outings.values() if o.split == 'train']
            test_outings = [o for o in outings.values() if o.split == 'test']
            
            if train_outings and test_outings:
                identity_outings[identity] = outings
        
        return identity_outings
    
    def _build_base_transform(self):
        """Build base preprocessing transform."""
        self.base_transform = A.Compose([
            A.LongestMaxSize(max_size=self.image_size),
            A.PadIfNeeded(
                min_height=self.image_size,
                min_width=self.image_size,
                border_mode=0,
            ),
            A.CenterCrop(height=self.image_size, width=self.image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    
    def _get_time_gap_bucket(self, days: int) -> str:
        """Get bucket label for time gap in days."""
        for min_days, max_days, label in TIME_GAP_BUCKETS:
            if min_days <= days < max_days:
                return label
        return TIME_GAP_BUCKETS[-1][2]  # Default to last bucket
    
    def _load_image(self, path: str) -> np.ndarray:
        """Load image as numpy array."""
        image = Image.open(path).convert('RGB')
        return np.array(image)
    
    def _apply_perturbation(
        self,
        image: np.ndarray,
        perturbation: str,
        param: float,
    ) -> torch.Tensor:
        """Apply a single perturbation and return preprocessed tensor."""
        perturbed = image.copy()
        
        if perturbation == 'rotation':
            transform = A.Affine(rotate=(param, param), p=1.0)
            perturbed = transform(image=perturbed)['image']
        
        elif perturbation == 'scale':
            transform = A.Affine(scale=(param, param), p=1.0)
            perturbed = transform(image=perturbed)['image']
        
        elif perturbation == 'brightness':
            mult = 1.0 + param
            transform = A.ColorJitter(
                brightness=(mult - 0.01, mult + 0.01),
                contrast=(1, 1), saturation=(1, 1), hue=(0, 0), p=1.0
            )
            perturbed = transform(image=perturbed)['image']
        
        elif perturbation == 'contrast':
            transform = A.ColorJitter(
                brightness=(1, 1), contrast=(param, param),
                saturation=(1, 1), hue=(0, 0), p=1.0
            )
            perturbed = transform(image=perturbed)['image']
        
        elif perturbation == 'saturation':
            transform = A.ColorJitter(
                brightness=(1, 1), contrast=(1, 1),
                saturation=(param, param), hue=(0, 0), p=1.0
            )
            perturbed = transform(image=perturbed)['image']
        
        elif perturbation == 'hue':
            transform = A.ColorJitter(
                brightness=(1, 1), contrast=(1, 1), saturation=(1, 1),
                hue=(param, param), p=1.0
            )
            perturbed = transform(image=perturbed)['image']
        
        elif perturbation == 'gaussian_blur':
            if param > 0:
                kernel = max(3, int(param * 4) | 1)
                transform = A.GaussianBlur(
                    blur_limit=(kernel, kernel),
                    sigma_limit=(param, param), p=1.0
                )
                perturbed = transform(image=perturbed)['image']
        
        elif perturbation == 'jpeg_quality':
            quality = int(param)
            if quality < 100:
                transform = A.ImageCompression(
                    quality_range=(quality, quality), p=1.0
                )
                perturbed = transform(image=perturbed)['image']
        
        elif perturbation == 'noise':
            if param > 0:
                # std_range expects standard deviation in 0-1 scale
                transform = A.GaussNoise(std_range=(param, param), p=1.0)
                perturbed = transform(image=perturbed)['image']
        
        # Apply base preprocessing
        result = self.base_transform(image=perturbed)['image']
        return result
    
    @torch.no_grad()
    def _extract_embedding(self, tensor: torch.Tensor) -> np.ndarray:
        """Extract embedding from preprocessed tensor."""
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        tensor = tensor.to(self.device)
        embedding = self.model(tensor, return_normalized=True)
        return embedding.cpu().numpy().squeeze()
    
    @torch.no_grad()
    def _extract_embeddings_batch(self, tensors: List[torch.Tensor]) -> np.ndarray:
        """Extract embeddings from a batch of tensors."""
        batch = torch.stack(tensors).to(self.device)
        embeddings = self.model(batch, return_normalized=True)
        return embeddings.cpu().numpy()
    
    def _compute_distance(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine distance."""
        return float(1.0 - np.dot(emb1, emb2))
    
    def analyze_outing_pair(
        self,
        identity: str,
        query_outing: OutingInfo,
        gallery_outing: OutingInfo,
        perturbations: Optional[List[str]] = None,
    ) -> OutingPairResult:
        """
        Analyze sensitivity for a single query-gallery outing pair.
        """
        if perturbations is None:
            perturbations = list(PERTURBATION_CONFIGS.keys())
        
        # Pick one image from each outing
        query_path = random.choice(query_outing.image_paths)
        gallery_path = random.choice(gallery_outing.image_paths)
        
        # Calculate time gap
        time_gap = abs((query_outing.date - gallery_outing.date).days)
        time_gap_bucket = self._get_time_gap_bucket(time_gap)
        
        # Load images
        query_image = self._load_image(query_path)
        gallery_image = self._load_image(gallery_path)
        
        # Get gallery embedding (fixed reference)
        gallery_tensor = self.base_transform(image=gallery_image)['image']
        gallery_emb = self._extract_embedding(gallery_tensor)
        
        # Get baseline distance
        query_tensor = self.base_transform(image=query_image)['image']
        query_emb = self._extract_embedding(query_tensor)
        baseline_distance = self._compute_distance(query_emb, gallery_emb)
        
        # Analyze each perturbation
        perturbation_results = {}
        sensitivity_ranges = {}
        
        for pert_name in perturbations:
            config = PERTURBATION_CONFIGS[pert_name]
            params = config['params']
            
            # Generate all perturbed tensors
            perturbed_tensors = []
            for param in params:
                tensor = self._apply_perturbation(query_image, pert_name, param)
                perturbed_tensors.append(tensor)
            
            # Batch extract embeddings
            embeddings = self._extract_embeddings_batch(perturbed_tensors)
            
            # Compute distances
            distances = np.array([
                self._compute_distance(emb, gallery_emb) 
                for emb in embeddings
            ])
            
            perturbation_results[pert_name] = {
                'params': params,
                'distances': distances,
            }
            sensitivity_ranges[pert_name] = float(distances.max() - distances.min())
        
        return OutingPairResult(
            identity=identity,
            query_outing_id=query_outing.outing_id,
            query_date=query_outing.date,
            query_path=query_path,
            gallery_outing_id=gallery_outing.outing_id,
            gallery_date=gallery_outing.date,
            gallery_path=gallery_path,
            time_gap_days=time_gap,
            time_gap_bucket=time_gap_bucket,
            baseline_distance=baseline_distance,
            perturbation_results=perturbation_results,
            sensitivity_ranges=sensitivity_ranges,
        )
    
    def analyze_identity(
        self,
        identity: str,
        max_pairs_per_identity: Optional[int] = 5,
        perturbations: Optional[List[str]] = None,
    ) -> IdentityTemporalResults:
        """
        Analyze all cross-outing pairs for a single identity.
        
        Args:
            identity: Identity to analyze
            max_pairs_per_identity: Max pairs to sample (None = all pairs)
            perturbations: Which perturbations to test
        """
        outings = self.identity_outings[identity]
        
        # Separate train and test outings
        train_outings = [o for o in outings.values() if o.split == 'train']
        test_outings = [o for o in outings.values() if o.split == 'test']
        
        # Generate all possible pairs (test query -> train gallery)
        all_pairs = []
        for query_outing in test_outings:
            for gallery_outing in train_outings:
                all_pairs.append((query_outing, gallery_outing))
        
        # Limit pairs if needed (unless max_pairs is None)
        if max_pairs_per_identity is not None and len(all_pairs) > max_pairs_per_identity:
            random.shuffle(all_pairs)
            all_pairs = all_pairs[:max_pairs_per_identity]
        
        # Analyze each pair
        outing_pair_results = []
        for query_outing, gallery_outing in all_pairs:
            try:
                result = self.analyze_outing_pair(
                    identity, query_outing, gallery_outing, perturbations
                )
                outing_pair_results.append(result)
            except Exception as e:
                print(f"\nWarning: Failed to analyze pair {query_outing.outing_id} -> "
                      f"{gallery_outing.outing_id}: {e}")
                continue
        
        if not outing_pair_results:
            raise ValueError(f"No valid pairs for identity {identity}")
        
        # Aggregate across pairs
        baselines = [r.baseline_distance for r in outing_pair_results]
        mean_baseline = float(np.mean(baselines))
        std_baseline = float(np.std(baselines))
        
        mean_sensitivity = {}
        std_sensitivity = {}
        for pert_name in PERTURBATION_CONFIGS.keys():
            ranges = [r.sensitivity_ranges.get(pert_name, 0) for r in outing_pair_results]
            mean_sensitivity[pert_name] = float(np.mean(ranges))
            std_sensitivity[pert_name] = float(np.std(ranges))
        
        return IdentityTemporalResults(
            identity=identity,
            outings=outings,
            outing_pairs=outing_pair_results,
            mean_baseline=mean_baseline,
            std_baseline=std_baseline,
            mean_sensitivity=mean_sensitivity,
            std_sensitivity=std_sensitivity,
        )
    
    def run(
        self,
        num_identities: Optional[int] = 30,
        max_pairs_per_identity: Optional[int] = 5,
        perturbations: Optional[List[str]] = None,
        analyze_all: bool = False,
    ) -> TemporalAnalysisResults:
        """
        Run temporal sensitivity analysis across multiple identities.
        
        Args:
            num_identities: Number of identities to sample (ignored if analyze_all=True)
            max_pairs_per_identity: Max pairs per identity (ignored if analyze_all=True)
            perturbations: Which perturbations to test (None = all)
            analyze_all: If True, analyze ALL identities with ALL outing pairs
        """
        if perturbations is None:
            perturbations = list(PERTURBATION_CONFIGS.keys())
        
        # Get identities to analyze
        random.seed(self.seed)
        all_identities = list(self.identity_outings.keys())
        
        if analyze_all:
            selected = all_identities
            max_pairs_per_identity = None  # No limit
            print(f"\nAnalyzing ALL {len(selected)} identities with ALL outing pairs...")
        else:
            random.shuffle(all_identities)
            selected = all_identities[:min(num_identities, len(all_identities))]
            print(f"\nAnalyzing {len(selected)} identities...")
        
        # Analyze each identity
        identity_results = []
        all_outing_pairs = []
        
        for identity in tqdm(selected, desc="Processing identities"):
            try:
                result = self.analyze_identity(
                    identity, max_pairs_per_identity, perturbations
                )
                identity_results.append(result)
                all_outing_pairs.extend(result.outing_pairs)
            except Exception as e:
                print(f"\nWarning: Failed to analyze {identity}: {e}")
                continue
        
        print(f"\nTotal outing pairs analyzed: {len(all_outing_pairs)}")
        
        # Aggregate by time gap
        print("Aggregating by time gap...")
        time_gap_aggregations = self._aggregate_by_time_gap(all_outing_pairs)
        
        # Overall aggregation
        all_baselines = [p.baseline_distance for p in all_outing_pairs]
        overall_sensitivity = {}
        for pert_name in perturbations:
            ranges = [p.sensitivity_ranges.get(pert_name, 0) for p in all_outing_pairs]
            overall_sensitivity[pert_name] = {
                'mean': float(np.mean(ranges)),
                'std': float(np.std(ranges)),
            }
        
        return TemporalAnalysisResults(
            num_identities=len(identity_results),
            num_outing_pairs=len(all_outing_pairs),
            checkpoint_path=self.checkpoint_path,
            data_root=str(self.data_root),
            identity_results=identity_results,
            all_outing_pairs=all_outing_pairs,
            time_gap_aggregations=time_gap_aggregations,
            overall_mean_baseline=float(np.mean(all_baselines)),
            overall_std_baseline=float(np.std(all_baselines)),
            overall_sensitivity=overall_sensitivity,
        )
    
    def _aggregate_by_time_gap(
        self,
        outing_pairs: List[OutingPairResult],
    ) -> Dict[str, Dict[str, Any]]:
        """Aggregate results by time gap buckets."""
        aggregations = {}
        
        for bucket_min, bucket_max, bucket_label in TIME_GAP_BUCKETS:
            # Filter pairs in this bucket
            bucket_pairs = [
                p for p in outing_pairs 
                if bucket_min <= p.time_gap_days < bucket_max
            ]
            
            if not bucket_pairs:
                continue
            
            # Compute statistics
            baselines = [p.baseline_distance for p in bucket_pairs]
            
            pert_stats = {}
            for pert_name in PERTURBATION_CONFIGS.keys():
                ranges = [p.sensitivity_ranges.get(pert_name, 0) for p in bucket_pairs]
                pert_stats[pert_name] = {
                    'mean_range': float(np.mean(ranges)),
                    'std_range': float(np.std(ranges)),
                }
            
            aggregations[bucket_label] = {
                'num_pairs': len(bucket_pairs),
                'mean_baseline': float(np.mean(baselines)),
                'std_baseline': float(np.std(baselines)),
                'perturbation_sensitivity': pert_stats,
            }
        
        return aggregations
    
    def save_results(
        self,
        results: TemporalAnalysisResults,
        output_dir: str,
    ) -> None:
        """Save results and generate visualizations."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving results to {output_dir}...")
        
        # Generate visualizations
        self._plot_time_gap_analysis(results, output_dir)
        self._plot_sensitivity_by_time_gap(results, output_dir)
        self._plot_sensitivity_ranking(results, output_dir)
        self._plot_per_identity_summary(results, output_dir)
        self._plot_baseline_vs_time_gap(results, output_dir)
        
        # Save numerical results
        self._save_outing_pairs_csv(results, output_dir)
        self._save_per_identity_csv(results, output_dir)
        self._save_time_gap_json(results, output_dir)
        self._save_summary_json(results, output_dir)
        
        # Print summary
        self._print_summary(results)
    
    def _plot_time_gap_analysis(
        self,
        results: TemporalAnalysisResults,
        output_dir: Path,
    ) -> None:
        """Plot baseline distance vs time gap."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Left: Scatter plot of baseline vs time gap
        ax = axes[0]
        time_gaps = [p.time_gap_days for p in results.all_outing_pairs]
        baselines = [p.baseline_distance for p in results.all_outing_pairs]
        
        scatter = ax.scatter(time_gaps, baselines, alpha=0.6, c=baselines, 
                            cmap='RdYlGn_r', edgecolors='white', linewidth=0.5)
        
        # Trend line
        z = np.polyfit(time_gaps, baselines, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(time_gaps), max(time_gaps), 100)
        ax.plot(x_line, p(x_line), 'r--', linewidth=2, alpha=0.7, 
                label=f'Trend (slope={z[0]:.2e})')
        
        ax.set_xlabel('Time Gap (days)', fontsize=11)
        ax.set_ylabel('Baseline Cosine Distance', fontsize=11)
        ax.set_title('Baseline Distance vs Time Between Encounters', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Distance')
        
        # Right: Box plot by time gap bucket
        ax = axes[1]
        bucket_labels = []
        bucket_baselines = []
        
        for bucket_label in [b[2] for b in TIME_GAP_BUCKETS]:
            if bucket_label in results.time_gap_aggregations:
                pairs_in_bucket = [
                    p.baseline_distance for p in results.all_outing_pairs
                    if p.time_gap_bucket == bucket_label
                ]
                if pairs_in_bucket:
                    bucket_labels.append(bucket_label)
                    bucket_baselines.append(pairs_in_bucket)
        
        if bucket_baselines:
            bp = ax.boxplot(bucket_baselines, tick_labels=bucket_labels, patch_artist=True)
            colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(bucket_baselines)))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        
        ax.set_xlabel('Time Gap Bucket', fontsize=11)
        ax.set_ylabel('Baseline Cosine Distance', fontsize=11)
        ax.set_title('Distance Distribution by Time Gap', fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', rotation=30)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig.savefig(output_dir / 'time_gap_analysis.png', dpi=150,
                    bbox_inches='tight', facecolor='white')
        plt.close(fig)
    
    def _plot_sensitivity_by_time_gap(
        self,
        results: TemporalAnalysisResults,
        output_dir: Path,
    ) -> None:
        """Plot sensitivity variation across time gap buckets."""
        # Get perturbations with enough data
        perturbations = list(PERTURBATION_CONFIGS.keys())
        n_perts = len(perturbations)
        n_cols = 3
        n_rows = (n_perts + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        axes = axes.flatten()
        
        bucket_order = [b[2] for b in TIME_GAP_BUCKETS]
        valid_buckets = [b for b in bucket_order if b in results.time_gap_aggregations]
        
        for i, pert_name in enumerate(perturbations):
            ax = axes[i]
            config = PERTURBATION_CONFIGS[pert_name]
            
            means = []
            stds = []
            x_labels = []
            
            for bucket_label in valid_buckets:
                agg = results.time_gap_aggregations[bucket_label]
                if pert_name in agg['perturbation_sensitivity']:
                    means.append(agg['perturbation_sensitivity'][pert_name]['mean_range'])
                    stds.append(agg['perturbation_sensitivity'][pert_name]['std_range'])
                    x_labels.append(bucket_label)
            
            if means:
                x_pos = np.arange(len(means))
                bars = ax.bar(x_pos, means, yerr=stds, capsize=3,
                              color=plt.cm.Set2(i % 8), edgecolor='white', linewidth=1)
                ax.set_xticks(x_pos)
                ax.set_xticklabels(x_labels, rotation=30, ha='right', fontsize=8)
            
            ax.set_ylabel('Sensitivity Range')
            ax.set_title(f"{config['name']}", fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        # Hide unused axes
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        
        plt.suptitle('Perturbation Sensitivity by Time Gap', fontsize=14, fontweight='bold')
        plt.tight_layout()
        fig.savefig(output_dir / 'sensitivity_by_time_gap.png', dpi=150,
                    bbox_inches='tight', facecolor='white')
        plt.close(fig)
    
    def _plot_sensitivity_ranking(
        self,
        results: TemporalAnalysisResults,
        output_dir: Path,
    ) -> None:
        """Create overall sensitivity ranking."""
        sorted_perts = sorted(
            results.overall_sensitivity.items(),
            key=lambda x: x[1]['mean'],
            reverse=True
        )
        
        names = [PERTURBATION_CONFIGS[p]['name'] for p, _ in sorted_perts]
        means = [s['mean'] for _, s in sorted_perts]
        stds = [s['std'] for _, s in sorted_perts]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        y_pos = np.arange(len(names))
        norm = np.array(means) / max(means) if max(means) > 0 else np.zeros_like(means)
        cmap = LinearSegmentedColormap.from_list('sens', ['#2ECC71', '#F1C40F', '#E74C3C'])
        colors = [cmap(n) for n in norm]
        
        bars = ax.barh(y_pos, means, xerr=stds, color=colors,
                       edgecolor='white', linewidth=1, capsize=3)
        
        ax.axvline(x=results.overall_mean_baseline, color='red', linestyle='--',
                   linewidth=2, label=f'Mean Baseline: {results.overall_mean_baseline:.4f}')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names)
        ax.set_xlabel('Mean Sensitivity Range ± Std')
        ax.set_title(f'Overall Perturbation Sensitivity\n({results.num_outing_pairs} outing pairs, '
                     f'{results.num_identities} identities)', fontsize=13, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(axis='x', alpha=0.3)
        
        for bar, val in zip(bars, means):
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                    f'{val:.4f}', va='center', fontsize=9)
        
        plt.tight_layout()
        fig.savefig(output_dir / 'sensitivity_ranking.png', dpi=150,
                    bbox_inches='tight', facecolor='white')
        plt.close(fig)
    
    def _plot_per_identity_summary(
        self,
        results: TemporalAnalysisResults,
        output_dir: Path,
    ) -> None:
        """Plot per-identity summary heatmap."""
        # Build matrix
        identities = [r.identity for r in results.identity_results]
        pert_names = list(PERTURBATION_CONFIGS.keys())
        
        matrix = np.zeros((len(identities), len(pert_names)))
        for i, result in enumerate(results.identity_results):
            for j, pert in enumerate(pert_names):
                matrix[i, j] = result.mean_sensitivity.get(pert, 0)
        
        fig, ax = plt.subplots(figsize=(12, max(8, len(identities) * 0.35)))
        
        im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn_r')
        
        ax.set_xticks(np.arange(len(pert_names)))
        ax.set_xticklabels([PERTURBATION_CONFIGS[p]['name'] for p in pert_names],
                          rotation=45, ha='right')
        ax.set_yticks(np.arange(len(identities)))
        
        # Truncate long identity names
        short_names = [id.split('__')[-1][:15] if '__' in id else id[:15] 
                       for id in identities]
        ax.set_yticklabels(short_names, fontsize=8)
        
        plt.colorbar(im, ax=ax, label='Mean Sensitivity Range')
        ax.set_title(f'Per-Identity Sensitivity\n({results.num_identities} identities)',
                     fontsize=13, fontweight='bold')
        
        plt.tight_layout()
        fig.savefig(output_dir / 'per_identity_heatmap.png', dpi=150,
                    bbox_inches='tight', facecolor='white')
        plt.close(fig)
    
    def _plot_baseline_vs_time_gap(
        self,
        results: TemporalAnalysisResults,
        output_dir: Path,
    ) -> None:
        """Detailed scatter with identity coloring."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Color by identity
        identities = list(set(p.identity for p in results.all_outing_pairs))
        identity_colors = {id: plt.cm.tab20(i % 20) for i, id in enumerate(identities)}
        
        for identity in identities:
            pairs = [p for p in results.all_outing_pairs if p.identity == identity]
            time_gaps = [p.time_gap_days for p in pairs]
            baselines = [p.baseline_distance for p in pairs]
            
            short_name = identity.split('__')[-1] if '__' in identity else identity
            ax.scatter(time_gaps, baselines, alpha=0.6, color=identity_colors[identity],
                       label=short_name[:12], edgecolors='white', linewidth=0.3, s=50)
        
        ax.set_xlabel('Time Gap (days)', fontsize=11)
        ax.set_ylabel('Baseline Cosine Distance', fontsize=11)
        ax.set_title('Baseline Distance by Identity and Time Gap', fontsize=12, fontweight='bold')
        
        # Legend outside
        if len(identities) <= 20:
            ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
        
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig.savefig(output_dir / 'baseline_by_identity.png', dpi=150,
                    bbox_inches='tight', facecolor='white')
        plt.close(fig)
    
    def _save_outing_pairs_csv(
        self,
        results: TemporalAnalysisResults,
        output_dir: Path,
    ) -> None:
        """Save detailed outing pair results."""
        rows = []
        for pair in results.all_outing_pairs:
            row = {
                'identity': pair.identity,
                'query_outing': pair.query_outing_id,
                'query_date': pair.query_date.strftime('%Y-%m-%d') if pair.query_date else '',
                'gallery_outing': pair.gallery_outing_id,
                'gallery_date': pair.gallery_date.strftime('%Y-%m-%d') if pair.gallery_date else '',
                'time_gap_days': pair.time_gap_days,
                'time_gap_bucket': pair.time_gap_bucket,
                'baseline_distance': pair.baseline_distance,
                'query_path': pair.query_path,
                'gallery_path': pair.gallery_path,
            }
            for pert_name, range_val in pair.sensitivity_ranges.items():
                row[f'{pert_name}_range'] = range_val
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_dir / 'outing_pairs.csv', index=False)
    
    def _save_per_identity_csv(
        self,
        results: TemporalAnalysisResults,
        output_dir: Path,
    ) -> None:
        """Save per-identity aggregated results."""
        rows = []
        for result in results.identity_results:
            row = {
                'identity': result.identity,
                'num_outing_pairs': len(result.outing_pairs),
                'num_train_outings': len([o for o in result.outings.values() if o.split == 'train']),
                'num_test_outings': len([o for o in result.outings.values() if o.split == 'test']),
                'mean_baseline': result.mean_baseline,
                'std_baseline': result.std_baseline,
            }
            for pert_name in PERTURBATION_CONFIGS.keys():
                row[f'{pert_name}_mean'] = result.mean_sensitivity.get(pert_name, 0)
                row[f'{pert_name}_std'] = result.std_sensitivity.get(pert_name, 0)
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_dir / 'per_identity_results.csv', index=False)
    
    def _save_time_gap_json(
        self,
        results: TemporalAnalysisResults,
        output_dir: Path,
    ) -> None:
        """Save time gap aggregation results."""
        with open(output_dir / 'time_gap_aggregations.json', 'w') as f:
            json.dump(results.time_gap_aggregations, f, indent=2)
    
    def _save_summary_json(
        self,
        results: TemporalAnalysisResults,
        output_dir: Path,
    ) -> None:
        """Save overall summary."""
        summary = {
            'num_identities': results.num_identities,
            'num_outing_pairs': results.num_outing_pairs,
            'checkpoint_path': results.checkpoint_path,
            'data_root': results.data_root,
            'overall_mean_baseline': results.overall_mean_baseline,
            'overall_std_baseline': results.overall_std_baseline,
            'sensitivity_ranking': [
                {
                    'perturbation': pert_name,
                    'display_name': PERTURBATION_CONFIGS[pert_name]['name'],
                    'mean_range': stats['mean'],
                    'std_range': stats['std'],
                }
                for pert_name, stats in sorted(
                    results.overall_sensitivity.items(),
                    key=lambda x: x[1]['mean'],
                    reverse=True
                )
            ],
        }
        
        with open(output_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
    
    def _print_summary(self, results: TemporalAnalysisResults) -> None:
        """Print summary to console."""
        print("\n" + "=" * 70)
        print("TEMPORAL SENSITIVITY ANALYSIS COMPLETE")
        print("=" * 70)
        print(f"Identities analyzed: {results.num_identities}")
        print(f"Outing pairs analyzed: {results.num_outing_pairs}")
        print(f"\nOverall Baseline Distance: {results.overall_mean_baseline:.4f} "
              f"± {results.overall_std_baseline:.4f}")
        
        print(f"\nBaseline by Time Gap:")
        for bucket_label, agg in results.time_gap_aggregations.items():
            print(f"  {bucket_label:15s}: {agg['mean_baseline']:.4f} ± {agg['std_baseline']:.4f} "
                  f"(n={agg['num_pairs']})")
        
        print(f"\nSensitivity Ranking (overall):")
        sorted_perts = sorted(results.overall_sensitivity.items(),
                              key=lambda x: x[1]['mean'], reverse=True)
        for i, (pert_name, stats) in enumerate(sorted_perts[:5]):
            display_name = PERTURBATION_CONFIGS[pert_name]['name']
            print(f"  {i+1:2d}. {display_name:20s} range={stats['mean']:.4f} ± {stats['std']:.4f}")


# Backward compatibility aliases
BatchSensitivityAnalyzer = TemporalSensitivityAnalyzer
EfficientBatchAnalyzer = TemporalSensitivityAnalyzer


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Temporal sensitivity analysis across outing pairs'
    )
    
    parser.add_argument('--checkpoint', '-c', required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data-root', '-d', required=True,
                        help='Path to star_dataset root')
    parser.add_argument('--output', '-o', default='./temporal_sensitivity_results',
                        help='Output directory')
    parser.add_argument('--num-identities', '-n', type=int, default=30,
                        help='Number of identities to analyze (default: 30, ignored if --all)')
    parser.add_argument('--max-pairs', '-m', type=int, default=5,
                        help='Max outing pairs per identity (default: 5, ignored if --all)')
    parser.add_argument('--all', '-a', action='store_true',
                        help='Analyze ALL identities with ALL outing pairs (overrides -n and -m)')
    parser.add_argument('--backbone', '-b', default='densenet121',
                        choices=['densenet121', 'densenet169', 'swinv2-tiny'],
                        help='Model backbone (default: densenet121)')
    parser.add_argument('--device', default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device (default: cuda)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--perturbations', '-p', nargs='+',
                        choices=list(PERTURBATION_CONFIGS.keys()),
                        help='Specific perturbations to test (default: all)')
    
    args = parser.parse_args()
    
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        return 1
    if not Path(args.data_root).exists():
        print(f"Error: Data root not found: {args.data_root}")
        return 1
    
    analyzer = TemporalSensitivityAnalyzer(
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        backbone=args.backbone,
        device=args.device,
        seed=args.seed,
    )
    
    results = analyzer.run(
        num_identities=args.num_identities,
        max_pairs_per_identity=args.max_pairs,
        perturbations=args.perturbations,
        analyze_all=getattr(args, 'all', False),
    )
    
    analyzer.save_results(results, args.output)
    
    print(f"\nResults saved to: {args.output}")
    return 0


if __name__ == '__main__':
    exit(main())

