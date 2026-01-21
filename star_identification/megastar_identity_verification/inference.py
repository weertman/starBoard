"""
Inference API for verification model.

Provides a simple interface for batch prediction on image pairs.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple, Optional, Union

import numpy as np
import torch
from PIL import Image

from .model import VerificationModel, create_verification_model
from .config import VerificationConfig
from .transforms import get_test_transforms

log = logging.getLogger("megastar_verification.inference")


class VerificationInference:
    """
    Lightweight inference wrapper for verification model.
    
    Handles model loading, image preprocessing, and batch prediction.
    
    Example:
        >>> inference = VerificationInference("checkpoints/best.pth")
        >>> prob = inference.predict_pair("img1.jpg", "img2.jpg")
        >>> print(f"P(same individual) = {prob:.3f}")
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        image_size: int = 224,
    ):
        """
        Initialize inference wrapper.
        
        Args:
            checkpoint_path: Path to model checkpoint (.pth file)
            device: Device to run inference on ("cuda" or "cpu")
            image_size: Input image size (must match training)
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.image_size = image_size
        self.checkpoint_path = checkpoint_path
        
        # Load model
        self.model = self._load_model(checkpoint_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Setup transforms
        self.transform = get_test_transforms(image_size)
        
        log.info(
            "VerificationInference initialized: device=%s, checkpoint=%s",
            self.device, Path(checkpoint_path).name
        )
    
    def _load_model(self, checkpoint_path: str) -> VerificationModel:
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        
        # Get config from checkpoint if available
        if "config" in checkpoint:
            config_dict = checkpoint["config"]
            config = VerificationConfig(
                backbone=config_dict.get("backbone", {}),
                cross_attention=config_dict.get("cross_attention", {}),
                classifier_hidden=config_dict.get("classifier_hidden", 256),
                classifier_dropout=config_dict.get("classifier_dropout", 0.3),
            )
        else:
            config = VerificationConfig()
        
        # Create model
        model = create_verification_model(config)
        
        # Load weights
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
        
        # Remove 'module.' prefix if present (from DataParallel)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        
        # Remap backbone keys if checkpoint uses ModuleList format (stages.X)
        # but model uses individual attributes (stageY where Y=X+1)
        state_dict = self._remap_backbone_keys(state_dict)
        
        model.load_state_dict(state_dict, strict=True)
        log.info("Loaded verification model from %s", checkpoint_path)
        
        return model
    
    def _remap_backbone_keys(self, state_dict: dict) -> dict:
        """
        Remap backbone state dict keys between different formats.
        
        Handles conversion from:
          backbone.stages.X.* -> backbone.stageY.* (where Y = X + 1)
          
        This allows loading checkpoints trained with nn.ModuleList backbone
        into models using individual stage attributes.
        """
        import re
        
        # Check if remapping is needed
        sample_keys = list(state_dict.keys())[:5]
        log.info("Sample checkpoint keys: %s", sample_keys)
        
        has_stages_list = any(k.startswith("backbone.stages.") for k in state_dict)
        has_stage_attrs = any(re.match(r"backbone\.stage\d+\.", k) for k in state_dict)
        
        log.info("Key format detection: has_stages_list=%s, has_stage_attrs=%s", 
                 has_stages_list, has_stage_attrs)
        
        if not has_stages_list:
            log.info("No remapping needed - keys already in correct format")
            return state_dict
        
        if has_stage_attrs and has_stages_list:
            # Mixed format: filter out the list-style keys, keep attribute-style
            log.warning("Mixed backbone key format detected, filtering out stages.* keys")
            filtered = {k: v for k, v in state_dict.items() 
                        if not k.startswith("backbone.stages.")}
            log.info("Filtered %d list-style keys, kept %d keys", 
                     len(state_dict) - len(filtered), len(filtered))
            return filtered
        
        log.info("Remapping %d keys from stages.X to stageY format", len(state_dict))
        
        remapped = {}
        stages_pattern = re.compile(r"^backbone\.stages\.(\d+)\.(.+)$")
        remapped_count = 0
        
        for key, value in state_dict.items():
            match = stages_pattern.match(key)
            if match:
                stage_idx = int(match.group(1))
                remainder = match.group(2)
                # stages.0 -> stage1, stages.1 -> stage2, etc.
                new_key = f"backbone.stage{stage_idx + 1}.{remainder}"
                remapped[new_key] = value
                remapped_count += 1
            else:
                remapped[key] = value
        
        log.info("Remapped %d backbone keys", remapped_count)
        sample_remapped = list(remapped.keys())[:5]
        log.info("Sample remapped keys: %s", sample_remapped)
        
        return remapped
    
    def _load_image(self, img_path: Union[str, Path]) -> torch.Tensor:
        """Load and preprocess a single image."""
        img = Image.open(img_path).convert("RGB")
        return self.transform(img)
    
    def _load_image_batch(
        self,
        paths_a: List[Union[str, Path]],
        paths_b: List[Union[str, Path]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load and preprocess a batch of image pairs."""
        imgs_a = torch.stack([self._load_image(p) for p in paths_a])
        imgs_b = torch.stack([self._load_image(p) for p in paths_b])
        return imgs_a, imgs_b
    
    @torch.no_grad()
    def predict_pair(
        self,
        img_a_path: Union[str, Path],
        img_b_path: Union[str, Path],
    ) -> float:
        """
        Get P(same individual) for a single image pair.
        
        Args:
            img_a_path: Path to first image
            img_b_path: Path to second image
            
        Returns:
            Probability that both images show the same individual (0.0 to 1.0)
        """
        img_a = self._load_image(img_a_path).unsqueeze(0).to(self.device)
        img_b = self._load_image(img_b_path).unsqueeze(0).to(self.device)
        
        prob = self.model.predict_proba(img_a, img_b)
        return prob.item()
    
    @torch.no_grad()
    def predict_batch(
        self,
        pairs: List[Tuple[Union[str, Path], Union[str, Path]]],
        batch_size: int = 16,
        progress_callback: Optional[callable] = None,
    ) -> np.ndarray:
        """
        Batch prediction for multiple image pairs.
        
        Args:
            pairs: List of (img_a_path, img_b_path) tuples
            batch_size: Number of pairs to process at once
            progress_callback: Optional callback(current, total) for progress updates
            
        Returns:
            Array of probabilities, shape (len(pairs),)
        """
        if not pairs:
            return np.array([])
        
        all_probs = []
        n_pairs = len(pairs)
        
        for start_idx in range(0, n_pairs, batch_size):
            end_idx = min(start_idx + batch_size, n_pairs)
            batch_pairs = pairs[start_idx:end_idx]
            
            # Separate paths
            paths_a = [p[0] for p in batch_pairs]
            paths_b = [p[1] for p in batch_pairs]
            
            # Load images
            imgs_a, imgs_b = self._load_image_batch(paths_a, paths_b)
            imgs_a = imgs_a.to(self.device)
            imgs_b = imgs_b.to(self.device)
            
            # Predict
            probs = self.model.predict_proba(imgs_a, imgs_b)
            all_probs.extend(probs.cpu().numpy().flatten())
            
            if progress_callback:
                progress_callback(end_idx, n_pairs)
        
        return np.array(all_probs)
    
    @torch.no_grad()
    def predict_matrix(
        self,
        query_paths: List[Union[str, Path]],
        gallery_paths: List[Union[str, Path]],
        batch_size: int = 16,
        progress_callback: Optional[callable] = None,
    ) -> np.ndarray:
        """
        Compute verification matrix for all query-gallery pairs.
        
        More efficient than predict_batch when you need all pairwise scores.
        
        Args:
            query_paths: List of query image paths
            gallery_paths: List of gallery image paths
            batch_size: Number of pairs to process at once
            progress_callback: Optional callback(current, total) for progress updates
            
        Returns:
            Matrix of probabilities, shape (len(query_paths), len(gallery_paths))
        """
        n_queries = len(query_paths)
        n_gallery = len(gallery_paths)
        total_pairs = n_queries * n_gallery
        
        # Build all pairs
        pairs = []
        for q_path in query_paths:
            for g_path in gallery_paths:
                pairs.append((q_path, g_path))
        
        # Predict all pairs
        probs = self.predict_batch(pairs, batch_size, progress_callback)
        
        # Reshape to matrix
        return probs.reshape(n_queries, n_gallery)


def load_inference(
    checkpoint_path: str,
    device: str = "cuda",
) -> VerificationInference:
    """
    Factory function to create VerificationInference instance.
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to run inference on
        
    Returns:
        VerificationInference instance
    """
    return VerificationInference(checkpoint_path, device)

