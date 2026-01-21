"""
Model utilities for MegaStarID.

Supports both legacy SwinReID and new MultiScaleReID architectures.
"""
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Dict, Any, Union

# Import models from wildlife_reid
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from wildlife_reid.models import SwinReID, MultiScaleReID, create_multiscale_model


def create_model(config) -> Union[SwinReID, MultiScaleReID]:
    """
    Create a fresh model from config.
    
    Automatically selects between legacy SwinReID and new MultiScaleReID
    based on config settings.
    
    Args:
        config: Config object with .model attribute
        
    Returns:
        Model instance (SwinReID or MultiScaleReID)
    """
    model_config = config.model
    
    # Check if using new MultiScaleReID features
    use_new_model = (
        hasattr(model_config, 'backbone') or
        hasattr(model_config, 'use_multiscale') or
        hasattr(model_config, 'use_bnneck')
    )
    
    if use_new_model:
        return create_multiscale_model(config)
    else:
        # Legacy path
        return SwinReID(
            model_name=model_config.name,
            embedding_dim=model_config.embedding_dim,
            dropout=model_config.dropout,
            pretrained=model_config.pretrained,
        )


def load_pretrained_model(
    config,
    checkpoint_path: str,
    device: torch.device,
    strict: bool = True,
) -> Union[SwinReID, MultiScaleReID]:
    """
    Load a model from a pre-trained checkpoint.
    
    Args:
        config: Model config
        checkpoint_path: Path to checkpoint file
        device: Device to load to
        strict: Whether to require exact state dict match
        
    Returns:
        Loaded model
    """
    model = create_model(config)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Handle DataParallel state dict
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    # Try to load, with helpful error message for architecture mismatch
    try:
        model.load_state_dict(state_dict, strict=strict)
    except RuntimeError as e:
        if not strict:
            raise
        # Try non-strict load and warn about missing/unexpected keys
        print(f"Warning: Strict loading failed, attempting non-strict load...")
        print(f"Original error: {e}")
        incompatible = model.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys:
            print(f"Missing keys: {incompatible.missing_keys[:5]}...")
        if incompatible.unexpected_keys:
            print(f"Unexpected keys: {incompatible.unexpected_keys[:5]}...")
    
    model = model.to(device)
    
    return model


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    
    return {
        'total': total,
        'trainable': trainable,
        'frozen': frozen,
    }


def freeze_backbone(model: Union[SwinReID, MultiScaleReID]):
    """
    Freeze the backbone, keep embedding head and other components trainable.
    
    Works with both SwinReID and MultiScaleReID.
    """
    # Find backbone - could be model.backbone or model.backbone.backbone
    backbone = None
    if hasattr(model, 'backbone'):
        backbone_attr = model.backbone
        # MultiScaleReID has backbone wrapper with .backbone inside
        if hasattr(backbone_attr, 'backbone'):
            backbone = backbone_attr.backbone
        else:
            backbone = backbone_attr
    
    if backbone is not None:
        for param in backbone.parameters():
            param.requires_grad = False
    
    # Keep these components trainable
    trainable_components = [
        'embedding_head', 'pool', 'gem_pool', 'attn_pool',
        'fusion', 'bnneck'
    ]
    
    for name in trainable_components:
        if hasattr(model, name):
            component = getattr(model, name)
            if component is not None:
                for param in component.parameters():
                    param.requires_grad = True


def unfreeze_backbone(model: nn.Module):
    """Unfreeze all parameters."""
    for param in model.parameters():
        param.requires_grad = True


def get_model_info(model: nn.Module) -> Dict[str, Any]:
    """
    Get detailed model information.
    
    Returns:
        Dict with model architecture details
    """
    params = count_parameters(model)
    
    info = {
        'total_params': params['total'],
        'trainable_params': params['trainable'],
        'frozen_params': params['frozen'],
        'total_params_millions': params['total'] / 1e6,
    }
    
    # Model-specific info
    if isinstance(model, MultiScaleReID):
        info['architecture'] = 'MultiScaleReID'
        info['backbone'] = model.backbone_name
        info['embedding_dim'] = model.embedding_dim
        info['use_multiscale'] = model.use_multiscale
        info['use_bnneck'] = model.use_bnneck
    elif isinstance(model, SwinReID):
        info['architecture'] = 'SwinReID'
        info['model_name'] = model.model_name
        info['embedding_dim'] = model.embedding_dim
    else:
        info['architecture'] = model.__class__.__name__
    
    return info
