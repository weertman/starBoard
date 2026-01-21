"""
Interpretability utilities for verification model.

Provides GradCAM visualization for backbone features and attention map
extraction from the cross-attention head.
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class VerificationGradCAM:
    """
    GradCAM for verification model backbone.
    
    Computes gradient-weighted class activation maps for both images
    in a verification pair, showing what regions contribute to the
    verification decision.
    
    Usage:
        gradcam = VerificationGradCAM(model)
        heatmap_a, heatmap_b = gradcam(img_a, img_b)
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_layer: str = 'backbone.stage4',
    ):
        """
        Args:
            model: VerificationModel instance
            target_layer: Layer to compute GradCAM for (default: final backbone stage)
        """
        self.model = model
        self.target_layer = target_layer
        
        # Storage for activations and gradients
        self._activations_a: Optional[torch.Tensor] = None
        self._activations_b: Optional[torch.Tensor] = None
        self._gradients_a: Optional[torch.Tensor] = None
        self._gradients_b: Optional[torch.Tensor] = None
        
        # Track which image is being processed
        self._processing_image_a = True
        
        # Get target module
        self._target_module = self._get_module(target_layer)
        
        # Register hooks
        self._forward_hook = self._target_module.register_forward_hook(self._forward_hook_fn)
        self._backward_hook = self._target_module.register_full_backward_hook(self._backward_hook_fn)
    
    def _get_module(self, layer_name: str) -> nn.Module:
        """Get module by name (e.g., 'backbone.stage4')."""
        module = self.model
        for part in layer_name.split('.'):
            module = getattr(module, part)
        return module
    
    def _forward_hook_fn(self, module: nn.Module, input: Tuple, output: torch.Tensor):
        """Capture activations during forward pass."""
        if self._processing_image_a:
            self._activations_a = output.detach()
        else:
            self._activations_b = output.detach()
    
    def _backward_hook_fn(self, module: nn.Module, grad_input: Tuple, grad_output: Tuple):
        """Capture gradients during backward pass."""
        # Gradients flow in reverse order, so B comes first in backward
        if self._gradients_b is None:
            self._gradients_b = grad_output[0].detach()
        else:
            self._gradients_a = grad_output[0].detach()
    
    def _compute_gradcam(
        self,
        activations: torch.Tensor,
        gradients: torch.Tensor,
    ) -> np.ndarray:
        """
        Compute GradCAM heatmap from activations and gradients.
        
        Args:
            activations: (B, C, H, W) feature activations
            gradients: (B, C, H, W) gradients
            
        Returns:
            (B, H, W) normalized heatmap in [0, 1]
        """
        # Global average pool gradients to get channel weights
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
        
        # Weighted combination of activation maps
        cam = (weights * activations).sum(dim=1)  # (B, H, W)
        
        # ReLU - only keep positive contributions
        cam = F.relu(cam)
        
        # Normalize per-sample
        batch_size = cam.shape[0]
        cam_flat = cam.view(batch_size, -1)
        cam_min = cam_flat.min(dim=1, keepdim=True)[0].unsqueeze(-1)
        cam_max = cam_flat.max(dim=1, keepdim=True)[0].unsqueeze(-1)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        
        return cam.cpu().numpy()
    
    def __call__(
        self,
        img_a: torch.Tensor,
        img_b: torch.Tensor,
        target_class: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute GradCAM heatmaps for both images in a pair.
        
        Args:
            img_a: (B, 3, H, W) first image
            img_b: (B, 3, H, W) second image
            target_class: Class to compute gradients for (1=same, 0=different)
            
        Returns:
            heatmap_a: (B, H_feat, W_feat) heatmap for image A
            heatmap_b: (B, H_feat, W_feat) heatmap for image B
        """
        self.model.eval()
        
        # Reset storage
        self._activations_a = None
        self._activations_b = None
        self._gradients_a = None
        self._gradients_b = None
        
        # Enable gradients for this computation
        img_a = img_a.requires_grad_(True)
        img_b = img_b.requires_grad_(True)
        
        # Forward pass - extract features separately to capture activations
        self._processing_image_a = True
        feat_a = self.model.extract_features(img_a)
        
        self._processing_image_a = False
        feat_b = self.model.extract_features(img_b)
        
        # Continue forward through cross-attention and classifier
        combined = self.model.cross_attention(feat_a, feat_b)
        logits = self.model.classifier(combined)
        
        # Backward pass
        self.model.zero_grad()
        
        # Create target tensor
        if target_class == 1:
            # Gradient w.r.t. "same individual" prediction
            target = logits.sum()
        else:
            # Gradient w.r.t. "different individual" prediction
            target = -logits.sum()
        
        target.backward()
        
        # Compute GradCAM heatmaps
        if self._activations_a is None or self._gradients_a is None:
            raise RuntimeError("Failed to capture activations/gradients for image A")
        if self._activations_b is None or self._gradients_b is None:
            raise RuntimeError("Failed to capture activations/gradients for image B")
        
        heatmap_a = self._compute_gradcam(self._activations_a, self._gradients_a)
        heatmap_b = self._compute_gradcam(self._activations_b, self._gradients_b)
        
        return heatmap_a, heatmap_b
    
    def remove_hooks(self):
        """Remove registered hooks."""
        self._forward_hook.remove()
        self._backward_hook.remove()
    
    def __del__(self):
        """Cleanup hooks on deletion."""
        try:
            self.remove_hooks()
        except:
            pass


class AttentionExtractor:
    """
    Extract cross-attention weights from verification model head.
    
    Captures attention maps showing which spatial positions in one image
    attend to which positions in the other image.
    
    Usage:
        extractor = AttentionExtractor(model)
        attention_maps = extractor(img_a, img_b)
        # attention_maps['layer0_a2b'] = (B, N, N) attention from A to B
    """
    
    def __init__(self, model: nn.Module):
        """
        Args:
            model: VerificationModel instance
        """
        self.model = model
        self._attention_weights: Dict[str, torch.Tensor] = {}
        self._hooks: List = []
        
        # Register hooks on all MultiheadAttention layers in cross_attention module
        self._register_attention_hooks()
    
    def _register_attention_hooks(self):
        """Register forward hooks on attention layers."""
        for layer_idx, layer in enumerate(self.model.cross_attention.layers):
            # Hook self-attention layers
            hook_a_self = layer.self_attn_a.register_forward_hook(
                self._make_attention_hook(f'layer{layer_idx}_self_a')
            )
            hook_b_self = layer.self_attn_b.register_forward_hook(
                self._make_attention_hook(f'layer{layer_idx}_self_b')
            )
            
            # Hook cross-attention layers (these are the most interesting)
            hook_a2b = layer.cross_attn_a2b.register_forward_hook(
                self._make_attention_hook(f'layer{layer_idx}_a2b')
            )
            hook_b2a = layer.cross_attn_b2a.register_forward_hook(
                self._make_attention_hook(f'layer{layer_idx}_b2a')
            )
            
            self._hooks.extend([hook_a_self, hook_b_self, hook_a2b, hook_b2a])
    
    def _make_attention_hook(self, name: str):
        """Create a hook function that captures attention weights."""
        def hook_fn(module: nn.Module, input: Tuple, output: Tuple):
            # MultiheadAttention returns (output, attention_weights) when need_weights=True
            # The output is a tuple (attn_output, attn_weights) if need_weights=True
            # But by default the model may not return weights
            # 
            # We manually compute attention weights from Q, K, V
            query, key, value = input[0], input[1], input[2]
            
            with torch.no_grad():
                # Get dimensions
                # The model uses batch_first=True, so input is (B, N, D)
                embed_dim = module.embed_dim
                num_heads = module.num_heads
                head_dim = embed_dim // num_heads
                batch_first = getattr(module, 'batch_first', False)
                
                # Get projections
                # Note: in_proj_weight combines W_q, W_k, W_v
                if module.in_proj_weight is not None:
                    # Combined projection
                    w_q, w_k, w_v = module.in_proj_weight.chunk(3)
                    b_q, b_k, b_v = (module.in_proj_bias.chunk(3) 
                                    if module.in_proj_bias is not None 
                                    else (None, None, None))
                else:
                    # Separate projections
                    w_q = module.q_proj_weight
                    w_k = module.k_proj_weight
                    w_v = module.v_proj_weight
                    b_q = module.q_proj_bias if hasattr(module, 'q_proj_bias') else None
                    b_k = module.k_proj_bias if hasattr(module, 'k_proj_bias') else None
                    b_v = module.v_proj_bias if hasattr(module, 'v_proj_bias') else None
                
                # Project Q and K
                q = torch.nn.functional.linear(query, w_q, b_q)
                k = torch.nn.functional.linear(key, w_k, b_k)
                
                # Reshape for multi-head attention
                if batch_first:
                    # Input: (B, L, E) -> (B, num_heads, L, head_dim)
                    B, L_q, _ = q.shape
                    L_k = k.shape[1]
                    
                    q = q.reshape(B, L_q, num_heads, head_dim).transpose(1, 2)
                    k = k.reshape(B, L_k, num_heads, head_dim).transpose(1, 2)
                else:
                    # Input: (L, B, E) -> (B, num_heads, L, head_dim)
                    L_q, B, _ = q.shape
                    L_k = k.shape[0]
                    
                    q = q.transpose(0, 1).reshape(B, L_q, num_heads, head_dim).transpose(1, 2)
                    k = k.transpose(0, 1).reshape(B, L_k, num_heads, head_dim).transpose(1, 2)
                
                # Compute attention scores
                scale = head_dim ** -0.5
                attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
                attn_weights = torch.softmax(attn_scores, dim=-1)
                
                # Average over heads: (B, num_heads, L_q, L_k) -> (B, L_q, L_k)
                attn_weights = attn_weights.mean(dim=1)
                
                self._attention_weights[name] = attn_weights.detach()
        
        return hook_fn
    
    def __call__(
        self,
        img_a: torch.Tensor,
        img_b: torch.Tensor,
    ) -> Dict[str, np.ndarray]:
        """
        Extract attention maps for an image pair.
        
        Args:
            img_a: (B, 3, H, W) first image
            img_b: (B, 3, H, W) second image
            
        Returns:
            Dict mapping attention layer names to attention weights:
            - 'layer{i}_a2b': (B, N, N) - positions in A attending to B
            - 'layer{i}_b2a': (B, N, N) - positions in B attending to A
            - 'layer{i}_self_a': (B, N, N) - self-attention in A
            - 'layer{i}_self_b': (B, N, N) - self-attention in B
            
            Where N = H_feat * W_feat (e.g., 7*7 = 49 for 224px input)
        """
        self.model.eval()
        self._attention_weights.clear()
        
        with torch.no_grad():
            # Forward pass - this triggers the hooks
            _ = self.model(img_a, img_b)
        
        # Convert to numpy
        result = {}
        for name, weights in self._attention_weights.items():
            result[name] = weights.cpu().numpy()
        
        return result
    
    def get_cross_attention_maps(
        self,
        img_a: torch.Tensor,
        img_b: torch.Tensor,
        layer_idx: int = -1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get just the cross-attention maps (A→B and B→A) for visualization.
        
        Args:
            img_a: (B, 3, H, W) first image
            img_b: (B, 3, H, W) second image
            layer_idx: Which layer to extract (-1 = last layer)
            
        Returns:
            attn_a2b: (B, N, N) attention from A positions to B positions
            attn_b2a: (B, N, N) attention from B positions to A positions
        """
        all_maps = self(img_a, img_b)
        
        num_layers = len(self.model.cross_attention.layers)
        if layer_idx < 0:
            layer_idx = num_layers + layer_idx
        
        attn_a2b = all_maps[f'layer{layer_idx}_a2b']
        attn_b2a = all_maps[f'layer{layer_idx}_b2a']
        
        return attn_a2b, attn_b2a
    
    def remove_hooks(self):
        """Remove registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
    
    def __del__(self):
        """Cleanup hooks on deletion."""
        try:
            self.remove_hooks()
        except:
            pass


def reshape_attention_to_spatial(
    attention: np.ndarray,
    feature_size: Tuple[int, int] = (7, 7),
) -> np.ndarray:
    """
    Reshape attention weights from (B, N, N) to spatial (B, H, W, H, W).
    
    Args:
        attention: (B, N, N) attention weights where N = H * W
        feature_size: (H, W) spatial dimensions of feature map
        
    Returns:
        (B, H_src, W_src, H_tgt, W_tgt) spatial attention
        For position (i, j) in source image, attention[:, i, j, :, :] shows
        the attention distribution over target image positions.
    """
    B, N1, N2 = attention.shape
    H, W = feature_size
    
    assert N1 == H * W, f"N1={N1} doesn't match H*W={H*W}"
    assert N2 == H * W, f"N2={N2} doesn't match H*W={H*W}"
    
    # Reshape: (B, N, N) -> (B, H, W, H, W)
    spatial_attn = attention.reshape(B, H, W, H, W)
    
    return spatial_attn


def get_attention_for_position(
    attention: np.ndarray,
    source_pos: Tuple[int, int],
    feature_size: Tuple[int, int] = (7, 7),
) -> np.ndarray:
    """
    Get attention map from a specific source position to all target positions.
    
    Args:
        attention: (B, N, N) attention weights
        source_pos: (row, col) position in source feature map
        feature_size: (H, W) spatial dimensions
        
    Returns:
        (B, H, W) attention distribution over target positions
    """
    spatial_attn = reshape_attention_to_spatial(attention, feature_size)
    row, col = source_pos
    
    # Extract attention from this source position to all target positions
    return spatial_attn[:, row, col, :, :]


def aggregate_attention_map(
    attention: np.ndarray,
    feature_size: Tuple[int, int] = (7, 7),
    aggregation: str = 'mean',
) -> np.ndarray:
    """
    Aggregate attention over all source positions.
    
    Args:
        attention: (B, N, N) attention weights
        feature_size: (H, W) spatial dimensions
        aggregation: 'mean', 'max', or 'sum'
        
    Returns:
        (B, H, W) aggregated attention map over target positions
    """
    spatial_attn = reshape_attention_to_spatial(attention, feature_size)
    # spatial_attn: (B, H_src, W_src, H_tgt, W_tgt)
    
    if aggregation == 'mean':
        # Average over source positions
        agg = spatial_attn.mean(axis=(1, 2))  # (B, H_tgt, W_tgt)
    elif aggregation == 'max':
        agg = spatial_attn.max(axis=(1, 2))
    elif aggregation == 'sum':
        agg = spatial_attn.sum(axis=(1, 2))
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")
    
    return agg


def compute_attention_entropy(
    attention: np.ndarray,
) -> np.ndarray:
    """
    Compute entropy of attention distribution (measure of focus/diffusion).
    
    Low entropy = focused attention (attending to few positions)
    High entropy = diffuse attention (attending to many positions)
    
    Args:
        attention: (B, N, N) attention weights (should sum to 1 along last dim)
        
    Returns:
        (B, N) entropy for each source position
    """
    # Add small epsilon to avoid log(0)
    attention = attention + 1e-10
    
    # Compute entropy: -sum(p * log(p))
    entropy = -np.sum(attention * np.log(attention), axis=-1)
    
    return entropy

