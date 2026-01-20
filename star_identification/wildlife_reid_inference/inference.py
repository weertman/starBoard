"""
Inference engine for Wildlife/Star Re-Identification.

Supports both legacy (src/) and new (megastarid/) model formats.
Updated to use megastarid as primary, with backward compatibility.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Union, List, Optional, Tuple, Any
from PIL import Image
import json
import sys
import threading

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from megastarid (primary)
from megastarid.models import create_model as megastarid_create_model, load_pretrained_model
from megastarid.config import MegaStarConfig, FinetuneConfig, ModelConfig
from megastarid.transforms import get_star_test_transforms, get_wildlife_test_transforms

# Handle both package and direct script imports
try:
    from .preprocessing import YOLOPreprocessor
except ImportError:
    from preprocessing import YOLOPreprocessor


# =============================================================================
# Legacy Config Support (for old checkpoints trained with src/)
# =============================================================================

class LegacyConfig:
    """
    Minimal config wrapper for loading old-style checkpoints.
    Mimics the interface of the old src.config.Config.
    """
    
    def __init__(
        self,
        model_name: str = 'microsoft/swinv2-base-patch4-window12to24-192to384-22kto1k-ft',
        embedding_dim: int = 768,
        dropout: float = 0.1,
        pretrained: bool = True,
        **kwargs  # Absorb other fields we don't need
    ):
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.pretrained = pretrained
        
        # Store any extra fields
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def get_model_image_size(self) -> int:
        """Get appropriate image size for the model"""
        model_lower = self.model_name.lower()
        
        if 'swinv2' in model_lower:
            if '256' in model_lower:
                return 256
            elif '384' in model_lower:
                return 384
            elif '192' in model_lower:
                return 192
            else:
                return 256
        elif 'swin' in model_lower:
            if '384' in model_lower:
                return 384
            elif '224' in model_lower:
                return 224
            else:
                return 224
        
        return 224
    
    @classmethod
    def load(cls, path: Path) -> 'LegacyConfig':
        """Load configuration from JSON"""
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Handle nested augmentation config if present (just ignore it for inference)
        if 'augmentation' in data and isinstance(data['augmentation'], dict):
            del data['augmentation']
        if 'pretrain_augmentation' in data:
            del data['pretrain_augmentation']
        if 'finetune_augmentation' in data:
            del data['finetune_augmentation']
        
        return cls(**data)


class LegacySwinReID(nn.Module):
    """
    Legacy SwinReID model for loading old checkpoints.
    Simplified version that matches the old src/models/swin_reid.py architecture.
    """
    
    def __init__(
        self,
        model_name: str,
        embedding_dim: int = 512,
        num_classes: int = 0,
        dropout: float = 0.0,
        pretrained: bool = True
    ):
        super().__init__()
        from transformers import SwinModel, Swinv2Model, AutoConfig
        
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.is_v2 = 'swinv2' in model_name.lower()
        
        # Load backbone
        if pretrained:
            if self.is_v2:
                self.backbone = Swinv2Model.from_pretrained(model_name)
            else:
                self.backbone = SwinModel.from_pretrained(model_name)
        else:
            config = AutoConfig.from_pretrained(model_name)
            if self.is_v2:
                self.backbone = Swinv2Model(config)
            else:
                self.backbone = SwinModel(config)
        
        self.hidden_dim = self.backbone.config.hidden_size
        
        # GeM pooling (simplified inline)
        self.gem_p = nn.Parameter(torch.ones(1) * 3.0)
        self.gem_eps = 1e-6
        
        # Embedding head (matches old architecture exactly)
        self.embedding = nn.Sequential(
            nn.Linear(self.hidden_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        )
        
        # Classifier (optional, for old checkpoints that have it)
        self.classifier = None
        if num_classes > 0:
            self.classifier = nn.Linear(embedding_dim, num_classes, bias=False)
    
    def forward(self, x, return_features=False, return_normalized=False):
        """Forward pass"""
        # Extract features
        outputs = self.backbone(x)
        features = outputs.last_hidden_state  # (B, L, D)
        
        # GeM pooling
        features = features.transpose(1, 2)  # (B, D, L)
        pooled = F.avg_pool1d(
            features.clamp(min=self.gem_eps).pow(self.gem_p),
            kernel_size=features.size(-1)
        ).pow(1. / self.gem_p).squeeze(-1)
        
        # Embed
        embeddings = self.embedding(pooled)
        
        if return_features or return_normalized:
            return F.normalize(embeddings, p=2, dim=1)
        
        if self.classifier is not None:
            logits = self.classifier(embeddings)
            return embeddings, logits
        
        return embeddings


def create_legacy_model(config: LegacyConfig, num_classes: int = 0) -> LegacySwinReID:
    """Create a legacy model from old-style config"""
    return LegacySwinReID(
        model_name=config.model_name,
        embedding_dim=config.embedding_dim,
        num_classes=num_classes,
        dropout=config.dropout,
        pretrained=config.pretrained
    )


def create_legacy_transforms(config: LegacyConfig):
    """Create test transforms compatible with legacy config"""
    from torchvision import transforms
    
    image_size = config.get_model_image_size()
    
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.14), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


# =============================================================================
# Config/Model Detection
# =============================================================================

def is_legacy_config(config_data: dict) -> bool:
    """Detect if config is old-style (src/) or new-style (megastarid/)"""
    # New configs have nested 'model' dict with 'backbone' key
    if 'model' in config_data and isinstance(config_data['model'], dict):
        if 'backbone' in config_data['model']:
            return False
    
    # Old configs have flat 'model_name' at top level
    if 'model_name' in config_data:
        return True
    
    # Default to legacy for safety
    return True


def is_legacy_checkpoint(checkpoint: dict) -> bool:
    """Detect if checkpoint is from old or new training system"""
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
    
    # New MultiScaleReID models have 'backbone.backbone' or 'fusion' keys
    for key in state_dict.keys():
        if 'backbone.backbone' in key or 'fusion' in key or 'bnneck' in key:
            return False
    
    # Old SwinReID models have direct 'backbone.' and 'embedding.'
    has_old_pattern = any(
        key.startswith('backbone.') or key.startswith('embedding.')
        for key in state_dict.keys()
    )
    
    return has_old_pattern


def infer_model_config_from_checkpoint(checkpoint: dict) -> dict:
    """
    Infer model configuration from checkpoint state dict.
    Useful when config file is missing.
    """
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
    
    config = {
        'backbone': 'densenet121',  # Default
        'embedding_dim': 512,
        'image_size': 384,
        'use_multiscale': True,
        'use_bnneck': True,
    }
    
    # Collect all keys for analysis
    keys_str = ' '.join(state_dict.keys()).lower()
    
    # Detect backbone type from state dict keys
    if 'denselayer' in keys_str or 'denseblock' in keys_str:
        # It's a DenseNet - distinguish 121 vs 169 by counting layers in stage4
        stage4_layers = sum(1 for k in state_dict.keys() if 'stage4' in k.lower() and 'denselayer' in k.lower() and 'norm1.weight' in k)
        if stage4_layers >= 32:
            config['backbone'] = 'densenet169'
        else:
            config['backbone'] = 'densenet121'
    elif 'layer4' in keys_str and 'layer1' in keys_str:
        # ResNet style
        config['backbone'] = 'resnet50'
    elif 'stages' in keys_str and 'dwconv' in keys_str:
        # ConvNeXt style
        config['backbone'] = 'convnext-tiny'
    elif 'swin' in keys_str or 'attn.logit_scale' in keys_str:
        # Swin Transformer
        config['backbone'] = 'swinv2-tiny'
    
    # Detect embedding dimension from embedding head
    for key, value in state_dict.items():
        if 'embedding_head' in key and 'weight' in key:
            if len(value.shape) == 2:
                config['embedding_dim'] = value.shape[0]
                break
    
    # Detect if bnneck is present
    config['use_bnneck'] = any('bnneck' in key for key in state_dict.keys())
    
    # Detect if multiscale is used (has fusion layer)
    config['use_multiscale'] = any('fusion' in key for key in state_dict.keys())
    
    return config


# =============================================================================
# Main Inference Engine
# =============================================================================

class WildlifeReIDInference:
    """
    Unified inference engine supporting both legacy and new model formats.
    Thread-safe for use in GUI applications.
    """

    def __init__(self, checkpoint_path: str, device: str = 'cuda', config_path: str = None):
        """
        Initialize inference engine.

        Args:
            checkpoint_path: Path to .pth checkpoint file
            device: 'cuda' or 'cpu'
            config_path: Path to config JSON (optional, will try to find automatically)
        """
        self._lock = threading.Lock()

        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        if device == 'cuda' and not torch.cuda.is_available():
            print("CUDA not available, using CPU")
            self.device = torch.device('cpu')

        self.checkpoint_path = Path(checkpoint_path)

        # Load checkpoint
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        # Detect checkpoint type
        self.is_legacy = is_legacy_checkpoint(checkpoint)
        print(f"Checkpoint type: {'legacy (src/)' if self.is_legacy else 'new (megastarid/)'}")

        # Load config
        config_data = None
        if config_path is not None:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
        elif 'config' in checkpoint:
            if isinstance(checkpoint['config'], dict):
                config_data = checkpoint['config']
            else:
                # Config object was pickled - extract its dict representation
                cfg_obj = checkpoint['config']
                if hasattr(cfg_obj, '__dict__'):
                    config_data = cfg_obj.__dict__
        
        if config_data is None:
            # Try to find config file
            checkpoint_dir = self.checkpoint_path.parent
            possible_configs = list(checkpoint_dir.glob("*config*.json"))
            if possible_configs:
                config_path = possible_configs[0]
                print(f"Found config file: {config_path}")
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
            else:
                # No config found - try to infer from checkpoint
                print("No config file found, inferring model configuration from checkpoint...")
                inferred = infer_model_config_from_checkpoint(checkpoint)
                print(f"Inferred config: {inferred}")
                
                # Create minimal config data
                config_data = {
                    'model': inferred,
                    'wildlife_root': './wildlifeReID_resized',
                    'star_dataset_root': './star_dataset_resized',
                    'checkpoint_dir': str(checkpoint_dir),
                    'batch_size': 32,
                }
                
                # Override is_legacy based on inference
                self.is_legacy = False

        # Create appropriate config and model
        if self.is_legacy:
            self._init_legacy(checkpoint, config_data)
        else:
            self._init_megastarid(checkpoint, config_data)

        # Optional preprocessor
        self.preprocessor = None
        print("Model loaded successfully!")

    def _init_legacy(self, checkpoint: dict, config_data: dict):
        """Initialize with legacy model/config"""
        self.config = LegacyConfig(**config_data)
        
        self.model_name = self.config.model_name
        self.embedding_dim = self.config.embedding_dim
        self.image_size = self.config.get_model_image_size()

        print(f"Legacy model info:")
        print(f"  Model: {self.model_name}")
        print(f"  Embedding dim: {self.embedding_dim}")
        print(f"  Image size: {self.image_size}")

        # Get number of classes from checkpoint
        num_classes = checkpoint.get('num_classes', 0)
        if num_classes == 0:
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            for key in state_dict:
                if 'classifier.weight' in key:
                    num_classes = state_dict[key].shape[0]
                    break

        # Create legacy model
        self.model = create_legacy_model(self.config, num_classes)

        # Load weights
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        if any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        self.model.load_state_dict(state_dict, strict=False)
        self.model = self.model.to(self.device)
        self.model.eval()

        # Create transforms
        self.transform = create_legacy_transforms(self.config)

    def _init_megastarid(self, checkpoint: dict, config_data: dict):
        """Initialize with megastarid model/config"""
        self.config = FinetuneConfig(**config_data)
        
        self.model_name = self.config.model.backbone
        self.embedding_dim = self.config.model.embedding_dim
        self.image_size = self.config.model.image_size

        print(f"MegaStarID model info:")
        print(f"  Backbone: {self.model_name}")
        print(f"  Embedding dim: {self.embedding_dim}")
        print(f"  Image size: {self.image_size}")

        # Load model
        self.model = load_pretrained_model(
            self.config,
            str(self.checkpoint_path),
            self.device,
            strict=False
        )
        self.model.eval()

        # Create transforms - try star transforms, fall back to wildlife transforms
        try:
            self.transform = get_star_test_transforms(self.image_size)
        except (TypeError, Exception) as e:
            print(f"Note: Using wildlife test transforms (star transforms unavailable: {e})")
            self.transform = get_wildlife_test_transforms(self.image_size)

    def set_preprocessor(self, yolo_model_path: str, confidence: float = 0.7, high_conf_threshold: float = 0.9):
        """Set YOLO preprocessor for automatic detection and cropping"""
        try:
            self.preprocessor = YOLOPreprocessor(yolo_model_path, confidence, high_conf_threshold)
            print("YOLO preprocessor initialized")
        except Exception as e:
            print(f"Warning: Failed to initialize preprocessor: {e}")
            self.preprocessor = None

    def embed_images(
        self,
        images: Union[str, List[str], Image.Image, List[Image.Image]],
        batch_size: int = 32,
        preprocess: bool = True
    ) -> np.ndarray:
        """Extract embeddings from images with thread safety"""
        with self._lock:
            if not isinstance(images, list):
                images = [images]

            # Preprocess with YOLO if requested and available
            if preprocess and self.preprocessor is not None:
                processed_images = []
                for img in images:
                    try:
                        processed = self.preprocessor.process_image(img)
                        if processed is not None:
                            processed_images.append(processed)
                        else:
                            if isinstance(img, str):
                                processed_images.append(Image.open(img).convert('RGB'))
                            else:
                                processed_images.append(img)
                    except Exception as e:
                        print(f"Warning: Preprocessing failed for image: {e}")
                        if isinstance(img, str):
                            processed_images.append(Image.open(img).convert('RGB'))
                        else:
                            processed_images.append(img)
                images = processed_images

            # Process in batches
            all_embeddings = []

            for i in range(0, len(images), batch_size):
                batch_images = images[i:i + batch_size]

                # Prepare batch
                batch_tensors = []
                for img in batch_images:
                    if isinstance(img, (str, Path)):
                        img = Image.open(img).convert('RGB')
                    elif not isinstance(img, Image.Image):
                        raise TypeError(f"Unsupported image type: {type(img)}")

                    img_tensor = self.transform(img)
                    batch_tensors.append(img_tensor)

                batch_tensor = torch.stack(batch_tensors).to(self.device)

                # Extract features
                with torch.no_grad():
                    if self.is_legacy:
                        embeddings = self.model(batch_tensor, return_features=True)
                    else:
                        embeddings = self.model(batch_tensor, return_normalized=True)

                all_embeddings.append(embeddings.cpu().numpy())

            return np.vstack(all_embeddings)

    def compute_similarity(
        self,
        embeddings1: np.ndarray,
        embeddings2: np.ndarray,
        metric: str = 'cosine'
    ) -> np.ndarray:
        """Compute pairwise similarities between embedding sets"""
        emb1 = torch.from_numpy(embeddings1).float()
        emb2 = torch.from_numpy(embeddings2).float()

        if metric == 'cosine':
            emb1 = F.normalize(emb1, p=2, dim=1)
            emb2 = F.normalize(emb2, p=2, dim=1)
            similarity = torch.mm(emb1, emb2.t())
        elif metric == 'euclidean':
            similarity = -torch.cdist(emb1, emb2, p=2)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        return similarity.numpy()

    def find_similar(
        self,
        query_images: Union[str, List[str], Image.Image, List[Image.Image]],
        gallery_images: List[Union[str, Image.Image]],
        top_k: int = 5,
        preprocess: bool = True
    ) -> List[List[Tuple[int, float]]]:
        """Find most similar images in gallery for each query"""
        if not isinstance(query_images, list):
            query_images = [query_images]

        query_embeddings = self.embed_images(query_images, preprocess=preprocess)
        gallery_embeddings = self.embed_images(gallery_images, preprocess=preprocess)

        similarities = self.compute_similarity(query_embeddings, gallery_embeddings)

        results = []
        for sim_row in similarities:
            top_indices = np.argsort(sim_row)[::-1][:top_k]
            top_scores = sim_row[top_indices]
            results.append([(int(idx), float(score)) for idx, score in zip(top_indices, top_scores)])

        return results
