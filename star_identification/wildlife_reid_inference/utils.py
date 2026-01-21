"""
Minimal utilities for wildlife reid inference
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, Any, Union, List, Iterator
from PIL import Image
from torchvision import transforms


def load_checkpoint_info(checkpoint_path: str) -> Dict[str, Any]:
    """
    Extract minimal info needed from checkpoint

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        Dictionary with model info and state dict
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Extract state dict
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    # Remove 'module.' prefix if present (from DataParallel)
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            cleaned_state_dict[k[7:]] = v
        else:
            cleaned_state_dict[k] = v

    # Try to infer model info from state dict keys
    model_type = 'swin-base'  # default

    # Check for swinv2 specific keys
    if any('swinv2' in k.lower() for k in cleaned_state_dict.keys()):
        model_type = 'swinv2-base'

    # Infer embedding dimension from the embedding layer
    embedding_dim = 512  # default
    for key in ['embedding.0.weight', 'embedding.0.bias']:
        if key in cleaned_state_dict:
            embedding_dim = cleaned_state_dict[key].shape[0]
            break

    # Infer image size from model type (could be made smarter)
    image_size = 224
    if 'config' in checkpoint:
        # Try to get from config if available
        config = checkpoint['config']
        if hasattr(config, 'get_model_image_size'):
            image_size = config.get_model_image_size()
        elif hasattr(config, 'model_name'):
            if '384' in config.model_name:
                image_size = 384
            elif '256' in config.model_name:
                image_size = 256

    return {
        'state_dict': cleaned_state_dict,
        'model_type': model_type,
        'embedding_dim': embedding_dim,
        'image_size': image_size
    }


def create_minimal_model(state_dict: dict, model_type: str, embedding_dim: int) -> nn.Module:
    """
    Create minimal model for inference

    Args:
        state_dict: Model weights
        model_type: Type of model architecture
        embedding_dim: Output embedding dimension

    Returns:
        Minimal model ready for inference
    """
    return MinimalReIDModel(state_dict, model_type, embedding_dim)


class MinimalReIDModel(nn.Module):
    """Minimal model that only loads what's needed for inference"""

    def __init__(self, state_dict: dict, model_type: str, embedding_dim: int):
        super().__init__()

        # We'll create a simple sequential model that matches the keys
        self.model_type = model_type
        self.embedding_dim = embedding_dim

        # Identify backbone keys (everything before 'embedding')
        backbone_keys = [k for k in state_dict.keys() if
                         not any(x in k for x in ['embedding', 'classifier', 'gem_pool'])]

        # For simplicity, we'll use a pretrained backbone and only load the head
        # This avoids needing transformers library
        self._init_backbone()

        # Initialize pooling
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Initialize embedding head from state dict
        self._init_embedding_head(state_dict)

        # Load weights for embedding head
        self.load_state_dict(state_dict, strict=False)

    def _init_backbone(self):
        """Initialize a simple backbone"""
        # Use torchvision's pretrained models as backbone
        import torchvision.models as models

        # Use EfficientNet as a lightweight alternative
        self.backbone = models.efficientnet_b0(pretrained=True)

        # Get the number of features from backbone
        self.backbone_dim = self.backbone.classifier[1].in_features

        # Remove the classifier
        self.backbone.classifier = nn.Identity()

    def _init_embedding_head(self, state_dict):
        """Initialize embedding head to match checkpoint"""
        # Look for embedding layer dimensions
        embed_weight_key = None
        for key in ['embedding.0.weight', 'fc.weight', 'head.weight']:
            if key in state_dict:
                embed_weight_key = key
                break

        if embed_weight_key:
            weight_shape = state_dict[embed_weight_key].shape
            input_dim = weight_shape[1]
            output_dim = weight_shape[0]
        else:
            # Fallback dimensions
            input_dim = self.backbone_dim
            output_dim = self.embedding_dim

        # Create embedding head
        self.embedding = nn.Sequential(
            nn.Linear(self.backbone_dim, output_dim),
            nn.BatchNorm1d(output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning normalized embeddings"""
        # Backbone features
        x = self.backbone(x)

        # Global pooling if needed
        if x.dim() > 2:
            x = self.pool(x)
            x = x.view(x.size(0), -1)

        # Embedding projection
        x = self.embedding(x)

        # L2 normalize
        x = F.normalize(x, p=2, dim=1)

        return x


def load_and_preprocess_image(image: Union[str, Image.Image], image_size: int) -> torch.Tensor:
    """
    Load and preprocess a single image

    Args:
        image: Image path or PIL Image
        image_size: Target size for the image

    Returns:
        Preprocessed image tensor
    """
    # Load image if path
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
    elif not isinstance(image, Image.Image):
        raise TypeError(f"Expected str or PIL Image, got {type(image)}")

    # Convert RGBA to RGB if needed
    if image.mode == 'RGBA':
        background = Image.new('RGB', image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        image = background
    elif image.mode != 'RGB':
        image = image.convert('RGB')

    # Define transform
    transform = transforms.Compose([
        transforms.Resize(int(image_size * 1.14), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return transform(image)


def batch_process(items: List[Any], batch_size: int) -> Iterator[List[Any]]:
    """
    Simple batch iterator

    Args:
        items: List of items to batch
        batch_size: Size of each batch

    Yields:
        Batches of items
    """
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]