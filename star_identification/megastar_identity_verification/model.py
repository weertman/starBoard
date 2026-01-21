"""
Verification Model with Cross-Attention.

Takes two images, extracts features using a shared backbone,
applies cross-attention between features, and outputs P(same individual).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, Any

from .config import VerificationConfig, BackboneConfig, CrossAttentionConfig


class CrossAttentionLayer(nn.Module):
    """Single cross-attention layer with self-attention and cross-attention."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # Self-attention for each branch
        self.self_attn_a = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.self_attn_b = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        
        # Cross-attention between branches
        self.cross_attn_a2b = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn_b2a = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        
        # Feed-forward networks
        self.ffn_a = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
        self.ffn_b = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
        
        # Layer norms
        self.norm_a1 = nn.LayerNorm(dim)
        self.norm_a2 = nn.LayerNorm(dim)
        self.norm_a3 = nn.LayerNorm(dim)
        self.norm_b1 = nn.LayerNorm(dim)
        self.norm_b2 = nn.LayerNorm(dim)
        self.norm_b3 = nn.LayerNorm(dim)
    
    def forward(
        self,
        feat_a: torch.Tensor,
        feat_b: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            feat_a: (B, N, D) features from image A
            feat_b: (B, N, D) features from image B
            
        Returns:
            Updated feat_a and feat_b with cross-attention applied
        """
        # Self-attention
        a_self, _ = self.self_attn_a(feat_a, feat_a, feat_a)
        feat_a = self.norm_a1(feat_a + a_self)
        
        b_self, _ = self.self_attn_b(feat_b, feat_b, feat_b)
        feat_b = self.norm_b1(feat_b + b_self)
        
        # Cross-attention: A attends to B, B attends to A
        a_cross, _ = self.cross_attn_a2b(feat_a, feat_b, feat_b)
        feat_a = self.norm_a2(feat_a + a_cross)
        
        b_cross, _ = self.cross_attn_b2a(feat_b, feat_a, feat_a)
        feat_b = self.norm_b2(feat_b + b_cross)
        
        # Feed-forward
        feat_a = self.norm_a3(feat_a + self.ffn_a(feat_a))
        feat_b = self.norm_b3(feat_b + self.ffn_b(feat_b))
        
        return feat_a, feat_b


class CrossAttentionModule(nn.Module):
    """Stack of cross-attention layers."""
    
    def __init__(self, config: CrossAttentionConfig):
        super().__init__()
        self.config = config
        
        # Project from backbone feature dim to hidden dim
        self.proj_a = nn.Linear(config.feature_dim, config.hidden_dim)
        self.proj_b = nn.Linear(config.feature_dim, config.hidden_dim)
        
        # Position embeddings (learnable)
        if config.use_pos_embed:
            # Will be resized dynamically based on feature map size
            self.pos_embed = nn.Parameter(torch.zeros(1, 196, config.hidden_dim))  # 14x14 default
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        else:
            self.pos_embed = None
        
        # Cross-attention layers
        self.layers = nn.ModuleList([
            CrossAttentionLayer(
                dim=config.hidden_dim,
                num_heads=config.num_heads,
                dropout=config.dropout,
            )
            for _ in range(config.num_layers)
        ])
        
        # Final norm
        self.norm = nn.LayerNorm(config.hidden_dim)
    
    def forward(
        self,
        feat_a: torch.Tensor,
        feat_b: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            feat_a: (B, C, H, W) feature map from image A
            feat_b: (B, C, H, W) feature map from image B
            
        Returns:
            (B, D) combined representation for classification
        """
        B, C, H, W = feat_a.shape
        
        # Flatten spatial dimensions: (B, C, H, W) -> (B, H*W, C)
        feat_a = feat_a.flatten(2).transpose(1, 2)  # (B, N, C) where N = H*W
        feat_b = feat_b.flatten(2).transpose(1, 2)
        
        # Project to hidden dim
        feat_a = self.proj_a(feat_a)  # (B, N, D)
        feat_b = self.proj_b(feat_b)
        
        # Add position embeddings
        if self.pos_embed is not None:
            N = feat_a.size(1)
            if N != self.pos_embed.size(1):
                # Interpolate position embeddings if feature map size differs
                pos = self.pos_embed.transpose(1, 2)  # (1, D, N_orig)
                pos = F.interpolate(pos, size=N, mode='linear', align_corners=False)
                pos = pos.transpose(1, 2)  # (1, N, D)
            else:
                pos = self.pos_embed
            feat_a = feat_a + pos
            feat_b = feat_b + pos
        
        # Apply cross-attention layers
        for layer in self.layers:
            feat_a, feat_b = layer(feat_a, feat_b)
        
        # Combine: concatenate mean-pooled features
        feat_a = self.norm(feat_a.mean(dim=1))  # (B, D)
        feat_b = self.norm(feat_b.mean(dim=1))  # (B, D)
        
        # Combine with element-wise operations
        combined = torch.cat([
            feat_a,
            feat_b,
            feat_a * feat_b,  # Element-wise product (similarity)
            torch.abs(feat_a - feat_b),  # Element-wise difference
        ], dim=1)  # (B, 4*D)
        
        return combined


def create_convnext_backbone(name: str = "convnext-small", pretrained: bool = True):
    """
    Create ConvNeXt backbone for feature extraction.
    
    Uses the same backbone structure as the embedding model (ConvNeXtBackbone from
    wildlife_reid) to ensure checkpoint compatibility.
    """
    # Import the same backbone class used by the embedding model
    # Try both import paths for flexibility
    try:
        from star_identification.wildlife_reid.models.multiscale_reid import ConvNeXtBackbone
    except ImportError:
        from wildlife_reid.models.multiscale_reid import ConvNeXtBackbone
    
    # Create backbone with matching structure
    backbone = ConvNeXtBackbone(
        model_name=name,
        pretrained=pretrained,
        extract_stages=(4,),  # Only need final stage for verification
    )
    
    return backbone


class VerificationModel(nn.Module):
    """
    Pairwise verification model.
    
    Takes two images, returns probability they show the same individual.
    """
    
    def __init__(self, config: VerificationConfig):
        super().__init__()
        self.config = config
        
        # Create backbone
        self.backbone = create_convnext_backbone(
            name=config.backbone.name,
            pretrained=True,
        )
        
        # Optionally freeze backbone
        if config.backbone.freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Cross-attention module
        self.cross_attention = CrossAttentionModule(config.cross_attention)
        
        # Classification head
        hidden_dim = config.cross_attention.hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 4, config.classifier_hidden),
            nn.ReLU(),
            nn.Dropout(config.classifier_dropout),
            nn.Linear(config.classifier_hidden, config.classifier_hidden),
            nn.ReLU(),
            nn.Dropout(config.classifier_dropout),
            nn.Linear(config.classifier_hidden, 1),  # Binary output
        )
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature map from single image."""
        _, final_features = self.backbone(x)
        return final_features  # (B, C, H, W)
    
    def forward(
        self,
        img_a: torch.Tensor,
        img_b: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            img_a: (B, 3, H, W) first image
            img_b: (B, 3, H, W) second image
            return_features: If True, also return intermediate features
            
        Returns:
            logits: (B, 1) raw logits (apply sigmoid for probability)
        """
        # Extract features from both images
        feat_a = self.extract_features(img_a)
        feat_b = self.extract_features(img_b)
        
        # Cross-attention
        combined = self.cross_attention(feat_a, feat_b)
        
        # Classification
        logits = self.classifier(combined)
        
        if return_features:
            return logits, {'feat_a': feat_a, 'feat_b': feat_b, 'combined': combined}
        
        return logits
    
    def predict_proba(
        self,
        img_a: torch.Tensor,
        img_b: torch.Tensor,
    ) -> torch.Tensor:
        """Get probability that images show same individual."""
        logits = self.forward(img_a, img_b)
        return torch.sigmoid(logits)
    
    def load_backbone_from_embedding_checkpoint(self, checkpoint_path: str):
        """
        Load backbone weights from a pre-trained embedding model checkpoint.
        
        Args:
            checkpoint_path: Path to embedding model checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint['model_state_dict']
        
        # Remove 'module.' prefix if present (from DataParallel)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        # Extract backbone weights (typically prefixed with 'backbone.')
        backbone_state = {}
        for k, v in state_dict.items():
            if k.startswith('backbone.'):
                # Remove 'backbone.' prefix
                new_key = k[9:]
                backbone_state[new_key] = v
        
        if backbone_state:
            # Load into our backbone
            missing, unexpected = self.backbone.load_state_dict(backbone_state, strict=False)
            print(f"Loaded backbone from {checkpoint_path}")
            if missing:
                print(f"  Missing keys: {len(missing)}")
            if unexpected:
                print(f"  Unexpected keys: {len(unexpected)}")
        else:
            print(f"Warning: No backbone weights found in {checkpoint_path}")
    
    def get_num_params(self) -> Dict[str, int]:
        """Get parameter counts."""
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        cross_attn_params = sum(p.numel() for p in self.cross_attention.parameters())
        classifier_params = sum(p.numel() for p in self.classifier.parameters())
        
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        
        return {
            'backbone': backbone_params,
            'cross_attention': cross_attn_params,
            'classifier': classifier_params,
            'trainable': trainable,
            'total': total,
        }


def create_verification_model(
    config: Optional[VerificationConfig] = None,
    backbone_checkpoint: Optional[str] = None,
) -> VerificationModel:
    """
    Factory function to create verification model.
    
    Args:
        config: Model configuration (uses defaults if None)
        backbone_checkpoint: Optional path to embedding model checkpoint
        
    Returns:
        VerificationModel instance
    """
    if config is None:
        config = VerificationConfig()
    
    model = VerificationModel(config)
    
    if backbone_checkpoint:
        model.load_backbone_from_embedding_checkpoint(backbone_checkpoint)
    
    return model

