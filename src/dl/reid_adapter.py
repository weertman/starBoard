"""
Adapter for MegaStarID inference.

Provides a clean interface between starBoard and the star_identification module.

Pipeline:
1. YOLO instance segmentation (starseg_best.pt) - crops/segments the star
2. Re-ID embedding (MegaStarID model) - generates embeddings from cropped star
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union
from contextlib import nullcontext

import numpy as np

from . import DL_AVAILABLE, DEVICE

log = logging.getLogger("starBoard.dl.reid_adapter")

# Add star_identification to path
_STAR_ID_PATH = Path(__file__).parent.parent.parent / "star_identification"
if str(_STAR_ID_PATH) not in sys.path:
    sys.path.insert(0, str(_STAR_ID_PATH))

# Default YOLO segmentation model path
DEFAULT_YOLO_MODEL = _STAR_ID_PATH / "wildlife_reid_inference" / "starseg_best.pt"


class ReIDAdapter:
    """
    Adapter for loading MegaStarID models and extracting embeddings.
    
    Handles:
    - YOLO instance segmentation preprocessing (star cropping)
    - Model loading with architecture auto-detection
    - CPU/GPU device management
    - Test-time augmentation (TTA)
    - Batch embedding extraction
    """
    
    def __init__(self):
        self._model = None
        self._model_path: Optional[str] = None
        self._device = None
        self._image_size: int = 384
        
        # YOLO preprocessor for star segmentation
        self._yolo_preprocessor = None
        self._yolo_available = False
    
    def is_model_loaded(self) -> bool:
        """Check if a model is currently loaded."""
        return self._model is not None
    
    def get_loaded_model_path(self) -> Optional[str]:
        """Get the path of the currently loaded model."""
        return self._model_path
    
    def is_yolo_available(self) -> bool:
        """Check if YOLO preprocessor is available."""
        return self._yolo_available
    
    def load_yolo_preprocessor(self, model_path: Optional[str] = None) -> bool:
        """
        Load the YOLO instance segmentation model for star cropping.
        
        Args:
            model_path: Path to YOLO model, defaults to starseg_best.pt
            
        Returns:
            True if successful, False otherwise
        """
        if model_path is None:
            model_path = str(DEFAULT_YOLO_MODEL)
        
        path = Path(model_path)
        if not path.exists():
            log.warning("YOLO model not found: %s", model_path)
            self._yolo_available = False
            return False
        
        try:
            from wildlife_reid_inference.preprocessing import YOLOPreprocessor, YOLO_AVAILABLE
            
            if not YOLO_AVAILABLE:
                log.warning("ultralytics not installed - YOLO preprocessing unavailable")
                self._yolo_available = False
                return False
            
            self._yolo_preprocessor = YOLOPreprocessor(
                str(path),
                confidence=0.7,
                high_conf_threshold=0.9
            )
            self._yolo_available = True
            log.info("Loaded YOLO preprocessor from %s", model_path)
            return True
            
        except ImportError as e:
            log.warning("Could not import YOLOPreprocessor: %s", e)
            self._yolo_available = False
            return False
        except Exception as e:
            log.error("Failed to load YOLO preprocessor: %s", e)
            self._yolo_available = False
            return False
    
    def preprocess_image(self, image_path: str) -> Optional['Image.Image']:
        """
        Preprocess a single image through YOLO segmentation.
        
        Args:
            image_path: Path to the raw image
            
        Returns:
            Cropped/segmented PIL Image, or None if detection failed
        """
        if not self._yolo_available or self._yolo_preprocessor is None:
            # If YOLO not available, just load the image directly
            from PIL import Image
            try:
                return Image.open(image_path).convert('RGB')
            except Exception as e:
                log.warning("Failed to load image %s: %s", image_path, e)
                return None
        
        try:
            result = self._yolo_preprocessor.process_image(image_path)
            return result
        except Exception as e:
            log.warning("YOLO preprocessing failed for %s: %s", image_path, e)
            return None
    
    def preprocess_batch(self, image_paths: List[str], batch_size: int = 8) -> List[Optional['Image.Image']]:
        """
        Preprocess a batch of images through YOLO segmentation.
        
        Args:
            image_paths: List of paths to raw images
            batch_size: Batch size for YOLO inference
            
        Returns:
            List of cropped/segmented PIL Images (None for failed detections)
        """
        if not self._yolo_available or self._yolo_preprocessor is None:
            # If YOLO not available, just load images directly
            from PIL import Image
            results = []
            for path in image_paths:
                try:
                    results.append(Image.open(path).convert('RGB'))
                except Exception:
                    results.append(None)
            return results
        
        try:
            return self._yolo_preprocessor.process_batch(image_paths, batch_size=batch_size)
        except Exception as e:
            log.error("YOLO batch preprocessing failed: %s", e)
            return [None] * len(image_paths)
    
    def load_model(self, checkpoint_path: str) -> bool:
        """
        Load a model from a checkpoint.
        
        Args:
            checkpoint_path: Path to the .pth checkpoint file
            
        Returns:
            True if successful, False otherwise
        """
        if not DL_AVAILABLE:
            log.error("Cannot load model: PyTorch not available")
            return False
        
        import torch
        
        path = Path(checkpoint_path)
        if not path.exists():
            log.error("Checkpoint not found: %s", checkpoint_path)
            return False
        
        try:
            # Set device
            self._device = torch.device(DEVICE)
            
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self._device, weights_only=False)
            
            # Extract config if present
            config = checkpoint.get('config', None)
            
            # Determine image size from config or use default
            if config and hasattr(config, 'model') and hasattr(config.model, 'image_size'):
                self._image_size = config.model.image_size
            else:
                self._image_size = 384
            
            # Create model
            model = self._create_model_from_checkpoint(checkpoint, config)
            
            if model is None:
                log.error("Failed to create model from checkpoint")
                return False
            
            # Load state dict
            state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
            
            # Handle DataParallel state dict
            if any(k.startswith('module.') for k in state_dict.keys()):
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            
            model.load_state_dict(state_dict, strict=False)
            model = model.to(self._device)
            model.eval()
            
            self._model = model
            self._model_path = str(path)
            
            log.info("Loaded model from %s (device=%s, image_size=%d)", 
                     checkpoint_path, DEVICE, self._image_size)
            return True
            
        except Exception as e:
            log.error("Failed to load model: %s", e)
            import traceback
            traceback.print_exc()
            return False
    
    def _infer_backbone_from_state_dict(self, state_dict: dict) -> str:
        """Infer the backbone architecture from state dict keys."""
        keys = list(state_dict.keys())
        
        # Check for ConvNeXt patterns (layer_scale is unique to ConvNeXt)
        if any('layer_scale' in k for k in keys):
            log.info("Detected ConvNeXt backbone from checkpoint keys")
            return "convnext-tiny"  # Default ConvNeXt variant
        
        # Check for DenseNet patterns
        if any('denselayer' in k for k in keys):
            log.info("Detected DenseNet backbone from checkpoint keys")
            return "densenet121"  # Default DenseNet variant
        
        # Check for SwinV2 patterns
        if any('swinv2' in k.lower() for k in keys) or any('layers.0.blocks' in k for k in keys):
            # Try to determine if it's tiny, small, or base from layer sizes
            log.info("Detected SwinV2 backbone from checkpoint keys")
            return "swinv2-tiny"  # Default to tiny
        
        # Check for ResNet patterns
        if any('layer1.0.conv1' in k for k in keys) and any('layer4' in k for k in keys):
            log.info("Detected ResNet backbone from checkpoint keys")
            return "resnet50"
        
        # Check for EfficientNet patterns
        if any('_blocks' in k for k in keys) or any('features.0' in k for k in keys):
            log.info("Detected EfficientNet backbone from checkpoint keys")
            return "efficientnet_b0"
        
        # Default fallback
        log.warning("Could not detect backbone from state dict, defaulting to swinv2-tiny")
        return "swinv2-tiny"
    
    def _infer_embedding_dim_from_state_dict(self, state_dict: dict) -> int:
        """Infer embedding dimension from state dict."""
        # Check embedding head output size
        for key in ['embedding_head.layer3.0.weight', 'embedding_head.layer2.0.weight', 
                    'embedding_head.layer1.0.weight', 'embedding.0.weight']:
            if key in state_dict:
                return state_dict[key].shape[0]
        
        # Check BN-neck
        if 'bnneck.0.weight' in state_dict:
            return state_dict['bnneck.0.weight'].shape[0]
        
        return 512  # Default
    
    def _create_model_from_checkpoint(self, checkpoint: dict, config) -> Optional['torch.nn.Module']:
        """Create a model instance from checkpoint metadata."""
        import torch.nn as nn
        
        # Get state dict to infer architecture
        state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
        
        # Handle DataParallel state dict
        if any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        try:
            # Try to import from megastarid
            from megastarid.models import create_model
            from megastarid.config import FinetuneConfig, ModelConfig
            
            if config is not None:
                # Use saved config
                model = create_model(config)
            else:
                # Infer architecture from state dict
                backbone = self._infer_backbone_from_state_dict(state_dict)
                embedding_dim = self._infer_embedding_dim_from_state_dict(state_dict)
                
                # Check if multiscale is used
                use_multiscale = any('fusion' in k for k in state_dict.keys())
                
                # Check if bnneck is used
                use_bnneck = any('bnneck' in k for k in state_dict.keys())
                
                log.info("Inferred model config: backbone=%s, embedding_dim=%d, multiscale=%s, bnneck=%s",
                        backbone, embedding_dim, use_multiscale, use_bnneck)
                
                model_config = ModelConfig(
                    backbone=backbone,
                    embedding_dim=embedding_dim,
                    use_multiscale=use_multiscale,
                    use_bnneck=use_bnneck,
                    image_size=self._image_size
                )
                config = FinetuneConfig(model=model_config)
                model = create_model(config)
            
            return model
            
        except ImportError as e:
            log.warning("Could not import megastarid.models: %s", e)
            
            # Fallback: try wildlife_reid directly
            try:
                from wildlife_reid.models import create_multiscale_model
                from megastarid.config import FinetuneConfig, ModelConfig
                
                backbone = self._infer_backbone_from_state_dict(state_dict)
                embedding_dim = self._infer_embedding_dim_from_state_dict(state_dict)
                use_multiscale = any('fusion' in k for k in state_dict.keys())
                use_bnneck = any('bnneck' in k for k in state_dict.keys())
                
                model_config = ModelConfig(
                    backbone=backbone,
                    embedding_dim=embedding_dim,
                    use_multiscale=use_multiscale,
                    use_bnneck=use_bnneck,
                    image_size=self._image_size
                )
                config = FinetuneConfig(model=model_config)
                model = create_multiscale_model(config)
                return model
                
            except Exception as e2:
                log.error("Fallback model creation failed: %s", e2)
                return None
    
    def unload_model(self):
        """Unload the current model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None
            self._model_path = None
            
            if DL_AVAILABLE:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            log.info("Unloaded model")
    
    def extract_embedding(self, image_path: str, use_tta: bool = True) -> Optional[np.ndarray]:
        """
        Extract embedding for a single image.
        
        Args:
            image_path: Path to the image file
            use_tta: Whether to use test-time augmentation (flip averaging)
            
        Returns:
            L2-normalized embedding vector of shape (embed_dim,), or None on failure
        """
        result = self.extract_batch([image_path], use_tta=use_tta)
        if result is not None and len(result) > 0:
            return result[0]
        return None
    
    def extract_batch(self, 
                      image_paths: List[str], 
                      use_tta: bool = True,
                      batch_size: int = 8,
                      use_yolo_preprocessing: bool = True,
                      use_horizontal_flip: bool = True,
                      use_vertical_flip: bool = True) -> Optional[np.ndarray]:
        """
        Extract embeddings for a batch of images.
        
        Pipeline:
        1. YOLO instance segmentation (if available and enabled) - crops/segments stars
        2. Transform and run through re-ID model
        3. Optional TTA (flip augmentation)
        
        Args:
            image_paths: List of image file paths
            use_tta: Whether to use test-time augmentation
            batch_size: Batch size for processing
            use_yolo_preprocessing: Whether to run YOLO segmentation first
            use_horizontal_flip: Include horizontal flip in TTA (faster to disable on CPU)
            use_vertical_flip: Include vertical flip in TTA (faster to disable on CPU)
            
        Returns:
            Array of shape (N, embed_dim) with L2-normalized embeddings, or None on failure
        """
        if not DL_AVAILABLE or self._model is None:
            return None
        
        if not image_paths:
            return np.array([])
        
        import torch
        import torch.nn.functional as F
        from PIL import Image
        from torchvision import transforms
        
        # Create transform
        transform = transforms.Compose([
            transforms.Resize((self._image_size, self._image_size), 
                            interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        all_embeddings = []
        
        try:
            self._model.eval()
            
            with torch.no_grad():
                for i in range(0, len(image_paths), batch_size):
                    batch_paths = image_paths[i:i + batch_size]
                    
                    # Step 1: YOLO preprocessing (if enabled)
                    if use_yolo_preprocessing and self._yolo_available:
                        preprocessed_images = self.preprocess_batch(batch_paths, batch_size=batch_size)
                    else:
                        # Load images directly
                        preprocessed_images = []
                        for path in batch_paths:
                            try:
                                preprocessed_images.append(Image.open(path).convert('RGB'))
                            except Exception:
                                preprocessed_images.append(None)
                    
                    # Step 2: Transform to tensors
                    batch_tensors = []
                    for img in preprocessed_images:
                        if img is not None:
                            try:
                                tensor = transform(img)
                                batch_tensors.append(tensor)
                            except Exception as e:
                                log.warning("Transform failed: %s", e)
                                batch_tensors.append(torch.zeros(3, self._image_size, self._image_size))
                        else:
                            # YOLO detection failed - use zero tensor as placeholder
                            batch_tensors.append(torch.zeros(3, self._image_size, self._image_size))
                    
                    if not batch_tensors:
                        continue
                    
                    images = torch.stack(batch_tensors).to(self._device)
                    
                    # Get autocast context
                    if self._device.type == 'cuda':
                        from torch.amp import autocast
                        ctx = autocast('cuda')
                    else:
                        ctx = nullcontext()
                    
                    with ctx:
                        # Original embedding
                        emb = self._model(images, return_normalized=True)
                        num_views = 1
                        
                        if use_tta:
                            if use_horizontal_flip:
                                # Horizontal flip
                                images_hflip = torch.flip(images, dims=[3])
                                emb_hflip = self._model(images_hflip, return_normalized=True)
                                emb = emb + emb_hflip
                                num_views += 1
                            
                            if use_vertical_flip:
                                # Vertical flip
                                images_vflip = torch.flip(images, dims=[2])
                                emb_vflip = self._model(images_vflip, return_normalized=True)
                                emb = emb + emb_vflip
                                num_views += 1
                            
                            # Average and re-normalize
                            if num_views > 1:
                                emb = emb / num_views
                                emb = F.normalize(emb, p=2, dim=1)
                    
                    all_embeddings.append(emb.cpu().numpy())
            
            if all_embeddings:
                return np.concatenate(all_embeddings, axis=0)
            return np.array([])
            
        except Exception as e:
            log.error("Embedding extraction failed: %s", e)
            import traceback
            traceback.print_exc()
            return None
    
    def get_embedding_dim(self) -> int:
        """Get the embedding dimension of the loaded model."""
        if self._model is not None and hasattr(self._model, 'embedding_dim'):
            return self._model.embedding_dim
        return 512  # Default
    
    def get_image_size(self) -> int:
        """Get the expected image size."""
        return self._image_size


# Global adapter instance
_adapter: Optional[ReIDAdapter] = None


def get_adapter() -> ReIDAdapter:
    """Get the global ReIDAdapter instance."""
    global _adapter
    if _adapter is None:
        _adapter = ReIDAdapter()
    return _adapter

