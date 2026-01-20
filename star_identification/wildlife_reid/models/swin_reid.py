"""
Swin Transformer V2 backbone for Re-Identification.

This is a copy of temporal_reid.models.swin_reid to keep wildlife_reid self-contained.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Swinv2Model, SwinModel, AutoConfig


class GeM(nn.Module):
    """
    Generalized Mean Pooling.
    
    GeM pooling generalizes average and max pooling:
    - p=1: average pooling
    - p→∞: max pooling
    - p=3 (typical): emphasizes larger activations
    """
    
    def __init__(self, p: float = 3.0, eps: float = 1e-6, learnable: bool = True):
        super().__init__()
        if learnable:
            self.p = nn.Parameter(torch.ones(1) * p)
        else:
            self.p = p
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, D, L) - batch, features, sequence length
        Returns:
            (B, D) - pooled features
        """
        p = self.p if isinstance(self.p, float) else self.p.clamp(min=1.0)
        return x.clamp(min=self.eps).pow(p).mean(dim=-1).pow(1.0 / p)


class SwinReID(nn.Module):
    """
    Swin Transformer for Re-Identification.
    
    Architecture:
        Input Image → Swin Backbone → GeM Pooling → Embedding Head → L2 Normalize
    
    The embedding head projects to a lower-dimensional space suitable for metric learning.
    """
    
    def __init__(
        self,
        model_name: str,
        embedding_dim: int = 512,
        dropout: float = 0.1,
        pretrained: bool = True,
    ):
        super().__init__()
        
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        
        # Detect model variant
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
        
        # Get backbone output dimension
        self.backbone_dim = self.backbone.config.hidden_size
        
        # Pooling
        self.pool = GeM(p=3.0, learnable=True)
        
        # Embedding head
        self.embedding_head = nn.Sequential(
            nn.Linear(self.backbone_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize embedding head weights."""
        for module in self.embedding_head:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, return_normalized: bool = True) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input images (B, C, H, W)
            return_normalized: If True, L2-normalize embeddings (default for inference)
        
        Returns:
            embeddings: (B, embedding_dim)
        """
        # Extract features from backbone
        outputs = self.backbone(x)
        features = outputs.last_hidden_state  # (B, L, D)
        
        # Pool across sequence dimension
        features = features.transpose(1, 2)  # (B, D, L)
        pooled = self.pool(features)  # (B, D)
        
        # Project to embedding space
        embeddings = self.embedding_head(pooled)  # (B, embedding_dim)
        
        if return_normalized:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    def get_backbone_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get raw backbone features before embedding head."""
        outputs = self.backbone(x)
        features = outputs.last_hidden_state
        features = features.transpose(1, 2)
        return self.pool(features)


def create_model(config) -> SwinReID:
    """
    Factory function to create model from config.
    
    Args:
        config: Config object with model settings
    
    Returns:
        SwinReID model
    """
    return SwinReID(
        model_name=config.model_name,
        embedding_dim=config.embedding_dim,
        dropout=config.dropout,
        pretrained=config.pretrained,
    )


