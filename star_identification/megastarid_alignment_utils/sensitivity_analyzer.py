"""
Embedding Distance Sensitivity Analyzer for MegaStarID.

Visualizes how input perturbations affect the distance between a query
and its known gallery match in the learned embedding space.

Usage:
    python -m megastarid_alignment_utils.sensitivity_analyzer \
        --query path/to/query.png \
        --gallery path/to/gallery_match.png \
        --checkpoint path/to/model.pth \
        --output sensitivity_results/

Or as a library:
    from megastarid_alignment_utils import SensitivityAnalyzer
    
    analyzer = SensitivityAnalyzer(checkpoint_path, device='cuda')
    results = analyzer.analyze(query_path, gallery_path)
    analyzer.plot_results(results, output_dir='./results')
"""

import argparse
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable, Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from megastarid.config import FinetuneConfig, ModelConfig
from megastarid.models import load_pretrained_model, create_model


class PerturbationFamily(Enum):
    """Categories of image perturbations to analyze."""
    ROTATION = "rotation"
    SCALE = "scale"
    TRANSLATION_X = "translation_x"
    TRANSLATION_Y = "translation_y"
    SHEAR = "shear"
    PERSPECTIVE = "perspective"
    BRIGHTNESS = "brightness"
    CONTRAST = "contrast"
    SATURATION = "saturation"
    HUE = "hue"
    GAUSSIAN_BLUR = "gaussian_blur"
    MOTION_BLUR = "motion_blur"
    GAUSSIAN_NOISE = "gaussian_noise"
    JPEG_COMPRESSION = "jpeg_compression"
    COARSE_DROPOUT = "coarse_dropout"
    HORIZONTAL_FLIP = "horizontal_flip"
    VERTICAL_FLIP = "vertical_flip"


@dataclass
class PerturbationConfig:
    """Configuration for a single perturbation sweep."""
    family: PerturbationFamily
    param_name: str
    param_range: np.ndarray
    display_name: str
    unit: str = ""
    is_discrete: bool = False


# Define sweep configurations for each perturbation family
PERTURBATION_CONFIGS: Dict[PerturbationFamily, PerturbationConfig] = {
    PerturbationFamily.ROTATION: PerturbationConfig(
        family=PerturbationFamily.ROTATION,
        param_name="angle",
        param_range=np.linspace(-180, 180, 37),  # Every 10 degrees
        display_name="Rotation",
        unit="°"
    ),
    PerturbationFamily.SCALE: PerturbationConfig(
        family=PerturbationFamily.SCALE,
        param_name="scale",
        param_range=np.linspace(0.5, 1.5, 21),
        display_name="Scale",
        unit="×"
    ),
    PerturbationFamily.TRANSLATION_X: PerturbationConfig(
        family=PerturbationFamily.TRANSLATION_X,
        param_name="translate_x",
        param_range=np.linspace(-0.3, 0.3, 21),
        display_name="Translation X",
        unit="(fraction)"
    ),
    PerturbationFamily.TRANSLATION_Y: PerturbationConfig(
        family=PerturbationFamily.TRANSLATION_Y,
        param_name="translate_y",
        param_range=np.linspace(-0.3, 0.3, 21),
        display_name="Translation Y",
        unit="(fraction)"
    ),
    PerturbationFamily.SHEAR: PerturbationConfig(
        family=PerturbationFamily.SHEAR,
        param_name="shear",
        param_range=np.linspace(-30, 30, 21),
        display_name="Shear",
        unit="°"
    ),
    PerturbationFamily.PERSPECTIVE: PerturbationConfig(
        family=PerturbationFamily.PERSPECTIVE,
        param_name="scale",
        param_range=np.linspace(0, 0.15, 16),
        display_name="Perspective Distortion",
        unit=""
    ),
    PerturbationFamily.BRIGHTNESS: PerturbationConfig(
        family=PerturbationFamily.BRIGHTNESS,
        param_name="brightness",
        param_range=np.linspace(-0.5, 0.5, 21),
        display_name="Brightness",
        unit=""
    ),
    PerturbationFamily.CONTRAST: PerturbationConfig(
        family=PerturbationFamily.CONTRAST,
        param_name="contrast",
        param_range=np.linspace(0.5, 1.5, 21),
        display_name="Contrast",
        unit="×"
    ),
    PerturbationFamily.SATURATION: PerturbationConfig(
        family=PerturbationFamily.SATURATION,
        param_name="saturation",
        param_range=np.linspace(0, 2.0, 21),
        display_name="Saturation",
        unit="×"
    ),
    PerturbationFamily.HUE: PerturbationConfig(
        family=PerturbationFamily.HUE,
        param_name="hue",
        param_range=np.linspace(-0.5, 0.5, 21),
        display_name="Hue Shift",
        unit=""
    ),
    PerturbationFamily.GAUSSIAN_BLUR: PerturbationConfig(
        family=PerturbationFamily.GAUSSIAN_BLUR,
        param_name="sigma",
        param_range=np.linspace(0, 10, 21),
        display_name="Gaussian Blur",
        unit="σ"
    ),
    PerturbationFamily.MOTION_BLUR: PerturbationConfig(
        family=PerturbationFamily.MOTION_BLUR,
        param_name="kernel_size",
        param_range=np.array([3, 5, 7, 9, 11, 13, 15, 17, 19, 21]),
        display_name="Motion Blur",
        unit="kernel",
        is_discrete=True
    ),
    PerturbationFamily.GAUSSIAN_NOISE: PerturbationConfig(
        family=PerturbationFamily.GAUSSIAN_NOISE,
        param_name="std",
        param_range=np.linspace(0, 0.15, 16),
        display_name="Gaussian Noise",
        unit="σ"
    ),
    PerturbationFamily.JPEG_COMPRESSION: PerturbationConfig(
        family=PerturbationFamily.JPEG_COMPRESSION,
        param_name="quality",
        param_range=np.array([100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 5]),
        display_name="JPEG Quality",
        unit="%",
        is_discrete=True
    ),
    PerturbationFamily.COARSE_DROPOUT: PerturbationConfig(
        family=PerturbationFamily.COARSE_DROPOUT,
        param_name="coverage",
        param_range=np.linspace(0, 0.4, 17),
        display_name="Occlusion Coverage",
        unit="%"
    ),
    PerturbationFamily.HORIZONTAL_FLIP: PerturbationConfig(
        family=PerturbationFamily.HORIZONTAL_FLIP,
        param_name="flip",
        param_range=np.array([0, 1]),
        display_name="Horizontal Flip",
        unit="",
        is_discrete=True
    ),
    PerturbationFamily.VERTICAL_FLIP: PerturbationConfig(
        family=PerturbationFamily.VERTICAL_FLIP,
        param_name="flip",
        param_range=np.array([0, 1]),
        display_name="Vertical Flip",
        unit="",
        is_discrete=True
    ),
}


def create_perturbation_transform(
    family: PerturbationFamily,
    param_value: float,
    image_size: int = 384,
) -> A.Compose:
    """
    Create an Albumentations transform for a specific perturbation.
    
    Args:
        family: Which perturbation family
        param_value: The parameter value for this perturbation
        image_size: Target image size
        
    Returns:
        Albumentations Compose transform
    """
    # Base preprocessing (resize and pad)
    base_transforms = [
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(
            min_height=image_size,
            min_width=image_size,
            border_mode=0,
        ),
        A.CenterCrop(height=image_size, width=image_size),
    ]
    
    # Add specific perturbation
    perturbation = []
    
    if family == PerturbationFamily.ROTATION:
        perturbation = [A.Affine(rotate=(param_value, param_value), p=1.0)]
    
    elif family == PerturbationFamily.SCALE:
        perturbation = [A.Affine(scale=(param_value, param_value), p=1.0)]
    
    elif family == PerturbationFamily.TRANSLATION_X:
        perturbation = [A.Affine(
            translate_percent={'x': (param_value, param_value), 'y': (0, 0)},
            p=1.0
        )]
    
    elif family == PerturbationFamily.TRANSLATION_Y:
        perturbation = [A.Affine(
            translate_percent={'x': (0, 0), 'y': (param_value, param_value)},
            p=1.0
        )]
    
    elif family == PerturbationFamily.SHEAR:
        perturbation = [A.Affine(shear=(param_value, param_value), p=1.0)]
    
    elif family == PerturbationFamily.PERSPECTIVE:
        if param_value > 0:
            perturbation = [A.Perspective(scale=(param_value, param_value), p=1.0)]
    
    elif family == PerturbationFamily.BRIGHTNESS:
        # ColorJitter brightness is multiplicative, we want additive-like behavior
        # Map [-0.5, 0.5] to [0.5, 1.5] brightness multiplier
        brightness_mult = 1.0 + param_value
        perturbation = [A.ColorJitter(
            brightness=(brightness_mult - 0.01, brightness_mult + 0.01),
            contrast=(1, 1), saturation=(1, 1), hue=(0, 0), p=1.0
        )]
    
    elif family == PerturbationFamily.CONTRAST:
        perturbation = [A.ColorJitter(
            brightness=(1, 1),
            contrast=(param_value, param_value),
            saturation=(1, 1), hue=(0, 0), p=1.0
        )]
    
    elif family == PerturbationFamily.SATURATION:
        perturbation = [A.ColorJitter(
            brightness=(1, 1), contrast=(1, 1),
            saturation=(param_value, param_value),
            hue=(0, 0), p=1.0
        )]
    
    elif family == PerturbationFamily.HUE:
        perturbation = [A.ColorJitter(
            brightness=(1, 1), contrast=(1, 1), saturation=(1, 1),
            hue=(param_value, param_value), p=1.0
        )]
    
    elif family == PerturbationFamily.GAUSSIAN_BLUR:
        if param_value > 0:
            # Kernel size must be odd
            kernel_size = max(3, int(param_value * 4) | 1)
            perturbation = [A.GaussianBlur(
                blur_limit=(kernel_size, kernel_size),
                sigma_limit=(param_value, param_value),
                p=1.0
            )]
    
    elif family == PerturbationFamily.MOTION_BLUR:
        kernel_size = int(param_value)
        if kernel_size >= 3:
            perturbation = [A.MotionBlur(blur_limit=(kernel_size, kernel_size), p=1.0)]
    
    elif family == PerturbationFamily.GAUSSIAN_NOISE:
        if param_value > 0:
            # std_range expects standard deviation in 0-1 scale
            perturbation = [A.GaussNoise(std_range=(param_value, param_value), p=1.0)]
    
    elif family == PerturbationFamily.JPEG_COMPRESSION:
        quality = int(param_value)
        if quality < 100:
            perturbation = [A.ImageCompression(quality_range=(quality, quality), p=1.0)]
    
    elif family == PerturbationFamily.COARSE_DROPOUT:
        if param_value > 0:
            # Calculate number of holes to approximate coverage
            hole_size = int(image_size * 0.1)  # 10% of image size per hole
            num_holes = max(1, int(param_value * 100 / 10))  # Rough approximation
            perturbation = [A.CoarseDropout(
                max_holes=num_holes, min_holes=num_holes,
                max_height=hole_size, min_height=hole_size,
                max_width=hole_size, min_width=hole_size,
                fill_value=0, p=1.0
            )]
    
    elif family == PerturbationFamily.HORIZONTAL_FLIP:
        if param_value > 0.5:
            perturbation = [A.HorizontalFlip(p=1.0)]
    
    elif family == PerturbationFamily.VERTICAL_FLIP:
        if param_value > 0.5:
            perturbation = [A.VerticalFlip(p=1.0)]
    
    # Normalization (ImageNet stats)
    normalize = [
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2(),
    ]
    
    return A.Compose(base_transforms + perturbation + normalize)


@dataclass
class SensitivityResult:
    """Results from a single perturbation sweep."""
    family: PerturbationFamily
    config: PerturbationConfig
    param_values: np.ndarray
    distances: np.ndarray
    baseline_distance: float
    perturbed_images: List[np.ndarray] = field(default_factory=list)  # For visualization


@dataclass
class AnalysisResults:
    """Complete sensitivity analysis results."""
    query_path: str
    gallery_path: str
    query_identity: Optional[str]
    baseline_distance: float
    results: Dict[PerturbationFamily, SensitivityResult]
    query_image: np.ndarray = None
    gallery_image: np.ndarray = None


class SensitivityAnalyzer:
    """
    Analyzes embedding distance sensitivity to input perturbations.
    
    Example:
        analyzer = SensitivityAnalyzer('checkpoint.pth', device='cuda')
        results = analyzer.analyze('query.png', 'gallery.png')
        analyzer.plot_results(results, 'output/')
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = 'cuda',
        image_size: int = 384,
        backbone: str = 'densenet121',
        config_override: Optional[Dict] = None,
    ):
        """
        Initialize the analyzer with a trained model.
        
        Args:
            checkpoint_path: Path to model checkpoint
            device: 'cuda' or 'cpu'
            image_size: Input image size
            backbone: Model backbone ('densenet121', 'densenet169', 'swinv2-tiny')
            config_override: Optional config overrides
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.image_size = image_size
        self.checkpoint_path = checkpoint_path
        self.backbone = backbone
        
        # Load model
        print(f"Loading model from {checkpoint_path}...")
        self.model = self._load_model(checkpoint_path, backbone, config_override)
        self.model.eval()
        print(f"Model loaded on {self.device} (backbone: {backbone})")
    
    def _load_model(self, checkpoint_path: str, backbone: str, config_override: Optional[Dict] = None):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Try to get config from checkpoint
        if 'config' in checkpoint:
            config_dict = checkpoint['config']
            config = FinetuneConfig(**config_dict)
        else:
            # Use default config with overrides
            config = FinetuneConfig()
            # Set backbone from argument
            config.model.backbone = backbone
            if config_override:
                for k, v in config_override.items():
                    if hasattr(config, k):
                        setattr(config, k, v)
                    elif hasattr(config.model, k):
                        setattr(config.model, k, v)
        
        # Update image size
        config.model.image_size = self.image_size
        
        # Load model
        model = load_pretrained_model(
            config=config,
            checkpoint_path=checkpoint_path,
            device=self.device,
            strict=False,
        )
        
        return model
    
    def _load_image(self, path: str) -> np.ndarray:
        """Load image as numpy array."""
        image = Image.open(path).convert('RGB')
        return np.array(image)
    
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Apply standard preprocessing (no perturbation)."""
        transform = create_perturbation_transform(
            PerturbationFamily.ROTATION, 0, self.image_size
        )
        result = transform(image=image)
        return result['image'].unsqueeze(0)
    
    def _extract_embedding(self, image_tensor: torch.Tensor) -> np.ndarray:
        """Extract L2-normalized embedding from image tensor."""
        with torch.no_grad():
            image_tensor = image_tensor.to(self.device)
            embedding = self.model(image_tensor, return_normalized=True)
            return embedding.cpu().numpy().squeeze()
    
    def _compute_distance(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine distance between two embeddings."""
        # Embeddings are already L2-normalized
        similarity = np.dot(emb1, emb2)
        return float(1.0 - similarity)
    
    def analyze_single_family(
        self,
        query_image: np.ndarray,
        gallery_embedding: np.ndarray,
        family: PerturbationFamily,
        store_images: bool = True,
        num_preview_images: int = 7,
    ) -> SensitivityResult:
        """
        Analyze sensitivity to a single perturbation family.
        
        Args:
            query_image: Query image as numpy array
            gallery_embedding: Pre-computed gallery embedding
            family: Which perturbation to analyze
            store_images: Whether to store perturbed images for visualization
            num_preview_images: Number of images to store for preview
        """
        config = PERTURBATION_CONFIGS[family]
        param_values = config.param_range
        distances = []
        preview_images = []
        
        # Select indices for preview images (evenly spaced)
        if store_images and len(param_values) > num_preview_images:
            preview_indices = np.linspace(0, len(param_values) - 1, num_preview_images, dtype=int)
        else:
            preview_indices = np.arange(len(param_values))
        
        for i, param_value in enumerate(param_values):
            # Create transform and apply
            transform = create_perturbation_transform(family, param_value, self.image_size)
            result = transform(image=query_image)
            image_tensor = result['image'].unsqueeze(0)
            
            # Extract embedding and compute distance
            embedding = self._extract_embedding(image_tensor)
            distance = self._compute_distance(embedding, gallery_embedding)
            distances.append(distance)
            
            # Store preview image (denormalized for visualization)
            if store_images and i in preview_indices:
                # Denormalize for visualization
                img_vis = result['image'].numpy().transpose(1, 2, 0)
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img_vis = img_vis * std + mean
                img_vis = np.clip(img_vis, 0, 1)
                preview_images.append(img_vis)
        
        # Compute baseline distance (no perturbation - use identity/neutral value)
        baseline_param = self._get_baseline_param(family)
        baseline_transform = create_perturbation_transform(family, baseline_param, self.image_size)
        baseline_result = baseline_transform(image=query_image)
        baseline_embedding = self._extract_embedding(baseline_result['image'].unsqueeze(0))
        baseline_distance = self._compute_distance(baseline_embedding, gallery_embedding)
        
        return SensitivityResult(
            family=family,
            config=config,
            param_values=param_values,
            distances=np.array(distances),
            baseline_distance=baseline_distance,
            perturbed_images=preview_images,
        )
    
    def _get_baseline_param(self, family: PerturbationFamily) -> float:
        """Get the 'identity' parameter value (no transformation)."""
        identity_params = {
            PerturbationFamily.ROTATION: 0,
            PerturbationFamily.SCALE: 1.0,
            PerturbationFamily.TRANSLATION_X: 0,
            PerturbationFamily.TRANSLATION_Y: 0,
            PerturbationFamily.SHEAR: 0,
            PerturbationFamily.PERSPECTIVE: 0,
            PerturbationFamily.BRIGHTNESS: 0,
            PerturbationFamily.CONTRAST: 1.0,
            PerturbationFamily.SATURATION: 1.0,
            PerturbationFamily.HUE: 0,
            PerturbationFamily.GAUSSIAN_BLUR: 0,
            PerturbationFamily.MOTION_BLUR: 0,
            PerturbationFamily.GAUSSIAN_NOISE: 0,
            PerturbationFamily.JPEG_COMPRESSION: 100,
            PerturbationFamily.COARSE_DROPOUT: 0,
            PerturbationFamily.HORIZONTAL_FLIP: 0,
            PerturbationFamily.VERTICAL_FLIP: 0,
        }
        return identity_params.get(family, 0)
    
    def analyze(
        self,
        query_path: str,
        gallery_path: str,
        families: Optional[List[PerturbationFamily]] = None,
        store_images: bool = True,
    ) -> AnalysisResults:
        """
        Run full sensitivity analysis.
        
        Args:
            query_path: Path to query image
            gallery_path: Path to gallery match image
            families: List of perturbation families to analyze (None = all)
            store_images: Whether to store perturbed images for visualization
            
        Returns:
            AnalysisResults with all sensitivity curves
        """
        if families is None:
            families = list(PerturbationFamily)
        
        print(f"\nAnalyzing sensitivity for {len(families)} perturbation families...")
        print(f"  Query: {query_path}")
        print(f"  Gallery: {gallery_path}")
        
        # Load images
        query_image = self._load_image(query_path)
        gallery_image = self._load_image(gallery_path)
        
        # Get gallery embedding (fixed reference)
        gallery_tensor = self._preprocess_image(gallery_image)
        gallery_embedding = self._extract_embedding(gallery_tensor)
        
        # Compute baseline distance
        query_tensor = self._preprocess_image(query_image)
        query_embedding = self._extract_embedding(query_tensor)
        baseline_distance = self._compute_distance(query_embedding, gallery_embedding)
        
        print(f"  Baseline distance: {baseline_distance:.4f}")
        
        # Analyze each family
        results = {}
        for family in families:
            print(f"  Analyzing {family.value}...", end=" ", flush=True)
            result = self.analyze_single_family(
                query_image, gallery_embedding, family, store_images
            )
            results[family] = result
            
            # Summary stats
            dist_range = result.distances.max() - result.distances.min()
            print(f"range: {dist_range:.4f}, max: {result.distances.max():.4f}")
        
        return AnalysisResults(
            query_path=query_path,
            gallery_path=gallery_path,
            query_identity=None,  # Can be set externally
            baseline_distance=baseline_distance,
            results=results,
            query_image=query_image,
            gallery_image=gallery_image,
        )
    
    def plot_results(
        self,
        results: AnalysisResults,
        output_dir: str,
        show_images: bool = True,
        figsize_per_family: Tuple[int, int] = (12, 4),
    ) -> None:
        """
        Generate visualization plots for sensitivity analysis results.
        
        Creates:
        - Individual sensitivity curves for each family
        - Summary heatmap
        - Combined overview plot
        
        Args:
            results: AnalysisResults from analyze()
            output_dir: Directory to save plots
            show_images: Whether to show perturbed image previews
            figsize_per_family: Figure size for individual family plots
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Custom color scheme
        plt.style.use('seaborn-v0_8-whitegrid')
        colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'accent': '#F18F01',
            'baseline': '#C73E1D',
            'background': '#F5F5F5',
        }
        
        # 1. Individual family plots
        for family, result in results.results.items():
            self._plot_single_family(result, results, output_dir, colors, show_images)
        
        # 2. Summary heatmap
        self._plot_summary_heatmap(results, output_dir, colors)
        
        # 3. Combined overview
        self._plot_combined_overview(results, output_dir, colors)
        
        # 4. Save numerical results
        self._save_numerical_results(results, output_dir)
        
        print(f"\nResults saved to {output_dir}")
    
    def _plot_single_family(
        self,
        result: SensitivityResult,
        analysis: AnalysisResults,
        output_dir: Path,
        colors: Dict,
        show_images: bool,
    ) -> None:
        """Plot sensitivity curve for a single perturbation family."""
        config = result.config
        
        if show_images and result.perturbed_images:
            # Create figure with image strip
            fig = plt.figure(figsize=(14, 6))
            gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2], hspace=0.3)
            
            # Image strip
            ax_images = fig.add_subplot(gs[0])
            n_images = len(result.perturbed_images)
            combined_width = n_images * result.perturbed_images[0].shape[1]
            combined_image = np.concatenate(result.perturbed_images, axis=1)
            ax_images.imshow(combined_image)
            ax_images.set_title(f"Query under {config.display_name} perturbation", fontsize=12)
            ax_images.axis('off')
            
            # Add parameter labels below each image
            img_width = result.perturbed_images[0].shape[1]
            preview_indices = np.linspace(0, len(result.param_values) - 1, n_images, dtype=int)
            for i, idx in enumerate(preview_indices):
                param = result.param_values[idx]
                if config.is_discrete:
                    label = f"{int(param)}{config.unit}"
                else:
                    label = f"{param:.2f}{config.unit}"
                ax_images.text(
                    (i + 0.5) * img_width, combined_image.shape[0] + 10,
                    label, ha='center', va='top', fontsize=9
                )
            
            ax_curve = fig.add_subplot(gs[1])
        else:
            fig, ax_curve = plt.subplots(figsize=(10, 5))
        
        # Plot sensitivity curve
        ax_curve.plot(
            result.param_values, result.distances,
            color=colors['primary'], linewidth=2.5, marker='o', markersize=4,
            label='Distance'
        )
        
        # Baseline reference line
        ax_curve.axhline(
            y=result.baseline_distance, color=colors['baseline'],
            linestyle='--', linewidth=1.5, alpha=0.7,
            label=f'Baseline ({result.baseline_distance:.4f})'
        )
        
        # Mark identity point
        baseline_param = self._get_baseline_param(result.family)
        ax_curve.axvline(
            x=baseline_param, color=colors['accent'],
            linestyle=':', linewidth=1.5, alpha=0.5,
            label='Identity transform'
        )
        
        # Formatting
        ax_curve.set_xlabel(f"{config.display_name} {config.unit}", fontsize=11)
        ax_curve.set_ylabel("Cosine Distance", fontsize=11)
        ax_curve.set_title(
            f"Distance Sensitivity to {config.display_name}\n"
            f"Query → Gallery Match",
            fontsize=13, fontweight='bold'
        )
        ax_curve.legend(loc='best', fontsize=9)
        ax_curve.grid(True, alpha=0.3)
        
        # Set y-axis to start from 0 for clarity
        y_max = max(result.distances.max() * 1.1, result.baseline_distance * 1.5)
        ax_curve.set_ylim(0, y_max)
        
        plt.tight_layout()
        
        filename = f"sensitivity_{result.family.value}.png"
        fig.savefig(output_dir / filename, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
    
    def _plot_summary_heatmap(
        self,
        results: AnalysisResults,
        output_dir: Path,
        colors: Dict,
    ) -> None:
        """Create summary heatmap showing sensitivity across all families."""
        families = list(results.results.keys())
        
        # Compute summary statistics for each family
        stats = {
            'Family': [],
            'Min Dist': [],
            'Max Dist': [],
            'Range': [],
            'Mean Dist': [],
            'Std Dist': [],
        }
        
        for family in families:
            result = results.results[family]
            stats['Family'].append(result.config.display_name)
            stats['Min Dist'].append(result.distances.min())
            stats['Max Dist'].append(result.distances.max())
            stats['Range'].append(result.distances.max() - result.distances.min())
            stats['Mean Dist'].append(result.distances.mean())
            stats['Std Dist'].append(result.distances.std())
        
        # Sort by range (most sensitive first)
        sorted_indices = np.argsort(stats['Range'])[::-1]
        
        fig, ax = plt.subplots(figsize=(10, max(6, len(families) * 0.4)))
        
        # Create horizontal bar chart
        y_pos = np.arange(len(families))
        ranges = [stats['Range'][i] for i in sorted_indices]
        names = [stats['Family'][i] for i in sorted_indices]
        
        # Color based on sensitivity (red = more sensitive)
        norm_ranges = np.array(ranges) / max(ranges) if max(ranges) > 0 else np.zeros_like(ranges)
        cmap = LinearSegmentedColormap.from_list('sensitivity', ['#2ECC71', '#F1C40F', '#E74C3C'])
        bar_colors = [cmap(r) for r in norm_ranges]
        
        bars = ax.barh(y_pos, ranges, color=bar_colors, edgecolor='white', linewidth=1)
        
        # Add baseline reference
        ax.axvline(x=results.baseline_distance, color=colors['baseline'],
                   linestyle='--', linewidth=2, label=f'Baseline: {results.baseline_distance:.4f}')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names)
        ax.set_xlabel('Distance Range (Max - Min)', fontsize=11)
        ax.set_title('Perturbation Sensitivity Summary\n(Higher = More Sensitive)', fontsize=13, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, ranges)):
            ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                    f'{val:.4f}', va='center', fontsize=9)
        
        plt.tight_layout()
        fig.savefig(output_dir / 'sensitivity_summary.png', dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
    
    def _plot_combined_overview(
        self,
        results: AnalysisResults,
        output_dir: Path,
        colors: Dict,
    ) -> None:
        """Create combined plot with all sensitivity curves."""
        # Group families by category
        categories = {
            'Geometric': [
                PerturbationFamily.ROTATION, PerturbationFamily.SCALE,
                PerturbationFamily.TRANSLATION_X, PerturbationFamily.TRANSLATION_Y,
                PerturbationFamily.SHEAR, PerturbationFamily.PERSPECTIVE,
            ],
            'Color': [
                PerturbationFamily.BRIGHTNESS, PerturbationFamily.CONTRAST,
                PerturbationFamily.SATURATION, PerturbationFamily.HUE,
            ],
            'Quality': [
                PerturbationFamily.GAUSSIAN_BLUR, PerturbationFamily.MOTION_BLUR,
                PerturbationFamily.GAUSSIAN_NOISE, PerturbationFamily.JPEG_COMPRESSION,
            ],
            'Occlusion': [
                PerturbationFamily.COARSE_DROPOUT,
                PerturbationFamily.HORIZONTAL_FLIP, PerturbationFamily.VERTICAL_FLIP,
            ],
        }
        
        category_colors = {
            'Geometric': '#3498DB',
            'Color': '#9B59B6',
            'Quality': '#E67E22',
            'Occlusion': '#1ABC9C',
        }
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for ax, (category, families) in zip(axes, categories.items()):
            for family in families:
                if family in results.results:
                    result = results.results[family]
                    
                    # Normalize x-axis for comparison
                    x_norm = np.linspace(0, 1, len(result.param_values))
                    
                    ax.plot(x_norm, result.distances,
                            label=result.config.display_name,
                            linewidth=1.5, alpha=0.8)
            
            ax.axhline(y=results.baseline_distance, color=colors['baseline'],
                       linestyle='--', linewidth=1.5, alpha=0.7, label='Baseline')
            
            ax.set_xlabel('Parameter (normalized)', fontsize=10)
            ax.set_ylabel('Cosine Distance', fontsize=10)
            ax.set_title(f'{category} Transforms', fontsize=12, fontweight='bold',
                        color=category_colors[category])
            ax.legend(fontsize=8, loc='best')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, None)
        
        plt.suptitle(
            f'Embedding Distance Sensitivity Analysis\n'
            f'Baseline Distance: {results.baseline_distance:.4f}',
            fontsize=14, fontweight='bold', y=1.02
        )
        
        plt.tight_layout()
        fig.savefig(output_dir / 'sensitivity_overview.png', dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
    
    def _save_numerical_results(
        self,
        results: AnalysisResults,
        output_dir: Path,
    ) -> None:
        """Save numerical results to JSON."""
        data = {
            'query_path': results.query_path,
            'gallery_path': results.gallery_path,
            'baseline_distance': results.baseline_distance,
            'families': {}
        }
        
        for family, result in results.results.items():
            data['families'][family.value] = {
                'display_name': result.config.display_name,
                'param_values': result.param_values.tolist(),
                'distances': result.distances.tolist(),
                'min_distance': float(result.distances.min()),
                'max_distance': float(result.distances.max()),
                'mean_distance': float(result.distances.mean()),
                'std_distance': float(result.distances.std()),
                'range': float(result.distances.max() - result.distances.min()),
            }
        
        with open(output_dir / 'sensitivity_results.json', 'w') as f:
            json.dump(data, f, indent=2)


def run_sensitivity_analysis(
    checkpoint_path: str,
    query_path: str,
    gallery_path: str,
    output_dir: str,
    device: str = 'cuda',
    backbone: str = 'densenet121',
    families: Optional[List[str]] = None,
) -> AnalysisResults:
    """
    Convenience function to run full analysis.
    
    Args:
        checkpoint_path: Path to model checkpoint
        query_path: Path to query image
        gallery_path: Path to gallery match image
        output_dir: Output directory for results
        device: 'cuda' or 'cpu'
        backbone: Model backbone ('densenet121', 'densenet169', 'swinv2-tiny')
        families: List of family names to analyze (None = all)
        
    Returns:
        AnalysisResults
    """
    analyzer = SensitivityAnalyzer(checkpoint_path, device=device, backbone=backbone)
    
    # Convert family names to enum if provided
    family_enums = None
    if families:
        family_enums = [PerturbationFamily(f) for f in families]
    
    results = analyzer.analyze(query_path, gallery_path, families=family_enums)
    analyzer.plot_results(results, output_dir)
    
    return results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Analyze embedding distance sensitivity to input perturbations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full analysis with all perturbations
  python -m megastarid_alignment_utils.sensitivity_analyzer \\
      --query star_dataset/folder1/star_001.png \\
      --gallery star_dataset/folder2/star_001.png \\
      --checkpoint checkpoints/megastarid/best.pth \\
      --output sensitivity_results/

  # Analyze only geometric transforms
  python -m megastarid_alignment_utils.sensitivity_analyzer \\
      --query query.png --gallery gallery.png \\
      --checkpoint model.pth --output results/ \\
      --families rotation scale translation_x translation_y
        """
    )
    
    parser.add_argument('--query', '-q', required=True,
                        help='Path to query image')
    parser.add_argument('--gallery', '-g', required=True,
                        help='Path to gallery match image (same identity)')
    parser.add_argument('--checkpoint', '-c', required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--output', '-o', default='./sensitivity_results',
                        help='Output directory (default: ./sensitivity_results)')
    parser.add_argument('--device', '-d', default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use (default: cuda)')
    parser.add_argument('--backbone', '-b', default='densenet121',
                        choices=['densenet121', 'densenet169', 'swinv2-tiny'],
                        help='Model backbone architecture (default: densenet121)')
    parser.add_argument('--families', '-f', nargs='+',
                        choices=[f.value for f in PerturbationFamily],
                        help='Specific perturbation families to analyze (default: all)')
    parser.add_argument('--no-images', action='store_true',
                        help='Skip storing preview images (faster)')
    parser.add_argument('--image-size', type=int, default=384,
                        help='Input image size (default: 384)')
    
    args = parser.parse_args()
    
    # Validate paths
    if not Path(args.query).exists():
        print(f"Error: Query image not found: {args.query}")
        return 1
    if not Path(args.gallery).exists():
        print(f"Error: Gallery image not found: {args.gallery}")
        return 1
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        return 1
    
    # Run analysis
    analyzer = SensitivityAnalyzer(
        args.checkpoint,
        device=args.device,
        image_size=args.image_size,
        backbone=args.backbone,
    )
    
    families = None
    if args.families:
        families = [PerturbationFamily(f) for f in args.families]
    
    results = analyzer.analyze(
        args.query,
        args.gallery,
        families=families,
        store_images=not args.no_images,
    )
    
    analyzer.plot_results(results, args.output, show_images=not args.no_images)
    
    print("\n" + "=" * 60)
    print("SENSITIVITY ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Baseline distance: {results.baseline_distance:.4f}")
    print(f"\nMost sensitive transforms (by range):")
    
    # Sort by sensitivity
    sorted_families = sorted(
        results.results.items(),
        key=lambda x: x[1].distances.max() - x[1].distances.min(),
        reverse=True
    )
    
    for i, (family, result) in enumerate(sorted_families[:5]):
        range_val = result.distances.max() - result.distances.min()
        print(f"  {i+1}. {result.config.display_name}: range={range_val:.4f}")
    
    print(f"\nResults saved to: {args.output}")
    
    return 0


if __name__ == '__main__':
    exit(main())

