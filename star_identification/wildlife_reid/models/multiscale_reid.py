"""
Multi-Scale Re-Identification Model.

Supports:
- Multiple backbones: SwinV2-Tiny, DenseNet-121/169, ResNet-50, ConvNeXt-Tiny
- Multi-scale feature extraction from intermediate stages
- Deep embedding head with optional residual connections
- BNNeck design pattern
- GeM pooling with optional attention pooling
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict
from collections import OrderedDict

# HuggingFace for SwinV2
from transformers import Swinv2Model, AutoConfig

# Torchvision for DenseNet
import torchvision.models as tv_models


class GeM(nn.Module):
    """
    Generalized Mean Pooling.
    
    p=1: average pooling
    p→∞: max pooling
    p=3 (typical): emphasizes larger activations
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
            x: (B, C, H, W) or (B, C, L) tensor
        Returns:
            (B, C) pooled features
        """
        p = self.p if isinstance(self.p, float) else self.p.clamp(min=1.0)
        
        if x.dim() == 4:
            # (B, C, H, W) -> (B, C)
            return x.clamp(min=self.eps).pow(p).mean(dim=(2, 3)).pow(1.0 / p)
        elif x.dim() == 3:
            # (B, C, L) -> (B, C)
            return x.clamp(min=self.eps).pow(p).mean(dim=-1).pow(1.0 / p)
        else:
            raise ValueError(f"Expected 3D or 4D input, got {x.dim()}D")


class AttentionPooling(nn.Module):
    """
    Learnable attention-based spatial pooling.
    
    Learns to weight spatial locations based on their relevance.
    """
    
    def __init__(self, in_channels: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 1, 1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            (B, C) attention-weighted pooled features
        """
        # Compute attention weights
        attn = self.attention(x)  # (B, 1, H, W)
        attn = attn.view(x.size(0), -1)  # (B, H*W)
        attn = F.softmax(attn, dim=-1)  # (B, H*W)
        attn = attn.view(x.size(0), 1, x.size(2), x.size(3))  # (B, 1, H, W)
        
        # Weighted sum
        out = (x * attn).sum(dim=(2, 3))  # (B, C)
        return out


class MultiScaleFeatureFusion(nn.Module):
    """
    Fuses features from multiple backbone stages.
    
    Upsamples all stages to the largest resolution, concatenates,
    and reduces channels with 1x1 conv.
    """
    
    def __init__(
        self,
        stage_channels: List[int],
        output_channels: int,
        target_size: Optional[Tuple[int, int]] = None,
    ):
        """
        Args:
            stage_channels: List of channel counts for each stage
            output_channels: Output channel count after fusion
            target_size: Target spatial size (H, W). If None, uses largest stage.
        """
        super().__init__()
        self.stage_channels = stage_channels
        self.target_size = target_size
        
        total_channels = sum(stage_channels)
        
        # 1x1 conv to reduce concatenated channels
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(total_channels, output_channels, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: List of (B, C_i, H_i, W_i) tensors from different stages
        Returns:
            (B, output_channels, H, W) fused features
        """
        if self.target_size is not None:
            target_h, target_w = self.target_size
        else:
            # Use the largest spatial size
            target_h = max(f.size(2) for f in features)
            target_w = max(f.size(3) for f in features)
        
        # Upsample all features to target size
        upsampled = []
        for feat in features:
            if feat.size(2) != target_h or feat.size(3) != target_w:
                feat = F.interpolate(
                    feat, size=(target_h, target_w),
                    mode='bilinear', align_corners=False
                )
            upsampled.append(feat)
        
        # Concatenate along channel dimension
        concat = torch.cat(upsampled, dim=1)
        
        # Reduce channels
        fused = self.fusion_conv(concat)
        
        return fused


class EmbeddingHead(nn.Module):
    """
    Deep embedding head with optional residual connections.
    
    Supports 2-layer or 3-layer configurations.
    """
    
    def __init__(
        self,
        in_dim: int,
        embedding_dim: int,
        depth: int = 3,
        dropout: float = 0.1,
        use_residual: bool = False,
    ):
        super().__init__()
        self.use_residual = use_residual
        self.depth = depth
        
        if depth == 2:
            self.layers = nn.Sequential(
                nn.Linear(in_dim, embedding_dim),
                nn.BatchNorm1d(embedding_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(embedding_dim, embedding_dim),
                nn.BatchNorm1d(embedding_dim),
            )
        elif depth == 3:
            hidden_dim = max(embedding_dim, in_dim // 2)
            
            self.layer1 = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            )
            self.layer2 = nn.Sequential(
                nn.Linear(hidden_dim, embedding_dim),
                nn.BatchNorm1d(embedding_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout / 2) if dropout > 0 else nn.Identity(),
            )
            self.layer3 = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.BatchNorm1d(embedding_dim),
            )
            
            # Residual projection if dimensions don't match
            if use_residual:
                if hidden_dim != embedding_dim:
                    self.res_proj = nn.Linear(hidden_dim, embedding_dim, bias=False)
                else:
                    self.res_proj = nn.Identity()
        else:
            raise ValueError(f"depth must be 2 or 3, got {depth}")
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.depth == 2:
            return self.layers(x)
        else:
            x1 = self.layer1(x)
            x2 = self.layer2(x1)
            x3 = self.layer3(x2)
            
            if self.use_residual:
                x3 = x3 + self.res_proj(x1)
            
            return x3


# =============================================================================
# Backbone Wrappers
# =============================================================================

class SwinV2Backbone(nn.Module):
    """
    SwinV2 backbone with multi-scale feature extraction.
    
    Extracts features from specified stages.
    
    Note: SwinV2's hidden_states are RAW features before LayerNorm, which can have
    very high variance (std ~10 for final stage). We apply LayerNorm to all stages
    to normalize feature magnitudes for proper multi-scale fusion.
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/swinv2-tiny-patch4-window8-256",
        pretrained: bool = True,
        extract_stages: Tuple[int, ...] = (2, 3, 4),
    ):
        super().__init__()
        self.extract_stages = extract_stages
        
        if pretrained:
            self.backbone = Swinv2Model.from_pretrained(model_name)
        else:
            config = AutoConfig.from_pretrained(model_name)
            self.backbone = Swinv2Model(config)
        
        # Get channel info from config
        self.hidden_size = self.backbone.config.hidden_size
        self.num_layers = len(self.backbone.config.depths)
        
        # Compute channel sizes for each stage
        # SwinV2 hidden states structure:
        #   hidden_states[0]: after patch embedding, channels = embed_dim
        #   hidden_states[1]: after stage 1, channels = embed_dim * 2
        #   hidden_states[2]: after stage 2, channels = embed_dim * 4
        #   hidden_states[3]: after stage 3, channels = embed_dim * 8
        #   hidden_states[4]: after stage 4, channels = embed_dim * 8 (same as 3)
        embed_dim = self.backbone.config.embed_dim
        self.stage_channels = []
        for stage_idx in extract_stages:
            if stage_idx == 0:
                ch = embed_dim
            elif stage_idx <= 3:
                ch = embed_dim * (2 ** stage_idx)
            else:
                # Stage 4 has same channels as stage 3
                ch = embed_dim * 8
            self.stage_channels.append(ch)
        
        # LayerNorm for each stage to normalize feature magnitudes
        # This is critical because raw hidden_states can have wildly different scales
        self.stage_norms = nn.ModuleList([
            nn.LayerNorm(ch) for ch in self.stage_channels
        ])
    
    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Args:
            x: (B, 3, H, W) input images
        Returns:
            features: List of (B, C_i, H_i, W_i) tensors for each extracted stage
            final: (B, hidden_size) final pooled features
        """
        # Get hidden states from all layers
        outputs = self.backbone(
            x,
            output_hidden_states=True,
            return_dict=True,
        )
        
        hidden_states = outputs.hidden_states  # Tuple of (B, L, C) for each stage
        # NOTE: hidden_states are RAW features BEFORE LayerNorm!
        # We apply our own LayerNorm to normalize all stages for proper fusion
        
        # Extract specified stages and reshape to spatial format
        features = []
        for i, stage_idx in enumerate(self.extract_stages):
            # hidden_states[0] is initial embedding, [1] is after stage 1, etc.
            hs = hidden_states[stage_idx]  # (B, L, C)
            
            # Apply LayerNorm to normalize features
            # This is critical for multi-scale fusion as raw hidden_states have wildly different scales
            hs = self.stage_norms[i](hs)  # (B, L, C) - normalized
            
            B, L, C = hs.shape
            
            # Compute spatial dimensions
            # After stage i, resolution is input_size / (4 * 2^(i-1)) for patch_size=4
            # We need to infer H, W from L
            H = W = int(L ** 0.5)
            
            # Reshape to (B, C, H, W)
            feat = hs.transpose(1, 2).reshape(B, C, H, W)
            features.append(feat)
        
        # Final features (use our normalized version)
        final = self.stage_norms[-1](hidden_states[self.extract_stages[-1]])  # (B, L, C)
        
        return features, final


class DenseNetBackbone(nn.Module):
    """
    DenseNet backbone with multi-scale feature extraction.
    
    Extracts features from dense blocks 2, 3, 4.
    """
    
    def __init__(
        self,
        model_name: str = "densenet121",
        pretrained: bool = True,
        extract_stages: Tuple[int, ...] = (2, 3, 4),
    ):
        super().__init__()
        self.extract_stages = extract_stages
        
        # Load pretrained DenseNet
        # Note: The stages extract features AFTER transition layers, not after denseblocks
        # For stage 4, we extract after denseblock4+norm5 (no transition)
        # DenseNet121: blocks=[6,12,24,16], growth_rate=32
        #   After transition1: 128, After transition2: 256, After transition3: 512, After norm5: 1024
        # DenseNet169: blocks=[6,12,32,32], growth_rate=32
        #   After transition1: 128, After transition2: 256, After transition3: 640, After norm5: 1664
        # DenseNet201: blocks=[6,12,48,32], growth_rate=32
        #   After transition1: 128, After transition2: 256, After transition3: 896, After norm5: 1920
        if model_name == "densenet121":
            weights = tv_models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = tv_models.densenet121(weights=weights)
            self.stage_channels_map = {1: 128, 2: 256, 3: 512, 4: 1024}
        elif model_name == "densenet169":
            weights = tv_models.DenseNet169_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = tv_models.densenet169(weights=weights)
            self.stage_channels_map = {1: 128, 2: 256, 3: 640, 4: 1664}
        elif model_name == "densenet201":
            weights = tv_models.DenseNet201_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = tv_models.densenet201(weights=weights)
            self.stage_channels_map = {1: 128, 2: 256, 3: 896, 4: 1920}
        else:
            raise ValueError(f"Unknown DenseNet variant: {model_name}")
        
        # Extract feature layers
        features = backbone.features
        
        # Build sequential stages
        # DenseNet structure: conv0, norm0, relu0, pool0, denseblock1, transition1, ...
        self.stem = nn.Sequential(
            features.conv0,
            features.norm0,
            features.relu0,
            features.pool0,
        )
        
        self.stage1 = nn.Sequential(
            features.denseblock1,
            features.transition1,
        )
        self.stage2 = nn.Sequential(
            features.denseblock2,
            features.transition2,
        )
        self.stage3 = nn.Sequential(
            features.denseblock3,
            features.transition3,
        )
        self.stage4 = nn.Sequential(
            features.denseblock4,
            features.norm5,  # Final batch norm
        )
        
        self.stages = [self.stage1, self.stage2, self.stage3, self.stage4]
        
        # Get output channels for extracted stages
        self.stage_channels = [self.stage_channels_map[s] for s in extract_stages]
        self.hidden_size = self.stage_channels_map[4]
    
    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Args:
            x: (B, 3, H, W) input images
        Returns:
            features: List of (B, C_i, H_i, W_i) tensors for each extracted stage
            final: (B, C, H, W) final features before pooling
        """
        x = self.stem(x)
        
        features = []
        for i, stage in enumerate(self.stages, start=1):
            x = stage(x)
            if i in self.extract_stages:
                features.append(x)
        
        return features, x


class ResNetBackbone(nn.Module):
    """
    ResNet backbone with multi-scale feature extraction.
    
    Extracts features from residual stages 2, 3, 4.
    ResNet-50 is the standard baseline in re-ID literature.
    """
    
    def __init__(
        self,
        model_name: str = "resnet50",
        pretrained: bool = True,
        extract_stages: Tuple[int, ...] = (2, 3, 4),
    ):
        super().__init__()
        self.extract_stages = extract_stages
        
        # Load pretrained ResNet
        # ResNet stage output channels:
        #   After layer1 (stage 1): 256
        #   After layer2 (stage 2): 512
        #   After layer3 (stage 3): 1024
        #   After layer4 (stage 4): 2048
        if model_name == "resnet50":
            weights = tv_models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            backbone = tv_models.resnet50(weights=weights)
            self.stage_channels_map = {1: 256, 2: 512, 3: 1024, 4: 2048}
        elif model_name == "resnet101":
            weights = tv_models.ResNet101_Weights.IMAGENET1K_V2 if pretrained else None
            backbone = tv_models.resnet101(weights=weights)
            self.stage_channels_map = {1: 256, 2: 512, 3: 1024, 4: 2048}
        else:
            raise ValueError(f"Unknown ResNet variant: {model_name}")
        
        # Build sequential stages
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        
        self.stage1 = backbone.layer1
        self.stage2 = backbone.layer2
        self.stage3 = backbone.layer3
        self.stage4 = backbone.layer4
        
        self.stages = [self.stage1, self.stage2, self.stage3, self.stage4]
        
        # Get output channels for extracted stages
        self.stage_channels = [self.stage_channels_map[s] for s in extract_stages]
        self.hidden_size = self.stage_channels_map[4]
    
    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Args:
            x: (B, 3, H, W) input images
        Returns:
            features: List of (B, C_i, H_i, W_i) tensors for each extracted stage
            final: (B, C, H, W) final features before pooling
        """
        x = self.stem(x)
        
        features = []
        for i, stage in enumerate(self.stages, start=1):
            x = stage(x)
            if i in self.extract_stages:
                features.append(x)
        
        return features, x


class ConvNeXtBackbone(nn.Module):
    """
    ConvNeXt backbone with multi-scale feature extraction.
    
    ConvNeXt modernizes CNNs with transformer-inspired design:
    - Large kernels (7x7)
    - Inverted bottleneck
    - Layer normalization
    - GELU activation
    
    Extracts features from stages 2, 3, 4.
    """
    
    def __init__(
        self,
        model_name: str = "convnext-tiny",
        pretrained: bool = True,
        extract_stages: Tuple[int, ...] = (2, 3, 4),
    ):
        super().__init__()
        self.extract_stages = extract_stages
        
        # Load pretrained ConvNeXt
        # ConvNeXt stage output channels:
        #   Stage 0 (stem + stage0): 96 (tiny) / 128 (small) / 128 (base)
        #   Stage 1: 192 (tiny) / 256 (small) / 256 (base)
        #   Stage 2: 384 (tiny) / 512 (small) / 512 (base)
        #   Stage 3: 768 (tiny) / 1024 (small) / 1024 (base)
        if model_name == "convnext-tiny":
            weights = tv_models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = tv_models.convnext_tiny(weights=weights)
            self.stage_channels_map = {1: 96, 2: 192, 3: 384, 4: 768}
        elif model_name == "convnext-small":
            weights = tv_models.ConvNeXt_Small_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = tv_models.convnext_small(weights=weights)
            self.stage_channels_map = {1: 96, 2: 192, 3: 384, 4: 768}
        elif model_name == "convnext-base":
            weights = tv_models.ConvNeXt_Base_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = tv_models.convnext_base(weights=weights)
            self.stage_channels_map = {1: 128, 2: 256, 3: 512, 4: 1024}
        else:
            raise ValueError(f"Unknown ConvNeXt variant: {model_name}")
        
        # ConvNeXt features structure:
        # features[0]: stem (conv + layernorm)
        # features[1]: stage 0 blocks
        # features[2]: downsample 1
        # features[3]: stage 1 blocks
        # features[4]: downsample 2
        # features[5]: stage 2 blocks
        # features[6]: downsample 3
        # features[7]: stage 3 blocks
        features = backbone.features
        
        # Build stages that match our extraction pattern
        # Stage 1 = stem + stage0
        self.stage1 = nn.Sequential(features[0], features[1])
        # Stage 2 = downsample1 + stage1
        self.stage2 = nn.Sequential(features[2], features[3])
        # Stage 3 = downsample2 + stage2
        self.stage3 = nn.Sequential(features[4], features[5])
        # Stage 4 = downsample3 + stage3
        self.stage4 = nn.Sequential(features[6], features[7])
        
        self.stages = [self.stage1, self.stage2, self.stage3, self.stage4]
        
        # Get output channels for extracted stages
        self.stage_channels = [self.stage_channels_map[s] for s in extract_stages]
        self.hidden_size = self.stage_channels_map[4]
    
    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Args:
            x: (B, 3, H, W) input images
        Returns:
            features: List of (B, C_i, H_i, W_i) tensors for each extracted stage
            final: (B, C, H, W) final features before pooling
        """
        features = []
        for i, stage in enumerate(self.stages, start=1):
            x = stage(x)
            if i in self.extract_stages:
                # ConvNeXt outputs (B, C, H, W) - already in correct format
                features.append(x)
        
        return features, x


# =============================================================================
# Main Model
# =============================================================================

class MultiScaleReID(nn.Module):
    """
    Multi-Scale Re-Identification Model.
    
    Architecture:
        Input → Backbone → Multi-Scale Fusion → Pooling → Embedding Head → BNNeck → L2 Norm
    
    Supports:
        - SwinV2-Tiny (transformer)
        - DenseNet-121, DenseNet-169 (dense CNN)
        - ResNet-50 (residual CNN, literature baseline)
        - ConvNeXt-Tiny (modern CNN)
        - Multi-scale feature fusion from intermediate stages
        - Deep embedding head (2 or 3 layers)
        - BNNeck for separate training/inference paths
        - GeM and optional attention pooling
    """
    
    def __init__(
        self,
        backbone: str = "swinv2-tiny",
        embedding_dim: int = 512,
        dropout: float = 0.1,
        pretrained: bool = True,
        use_multiscale: bool = True,
        multiscale_stages: Tuple[int, ...] = (2, 3, 4),
        embedding_head_depth: int = 3,
        use_residual_head: bool = False,
        use_attention_pooling: bool = False,
        use_bnneck: bool = True,
    ):
        super().__init__()
        
        self.backbone_name = backbone
        self.embedding_dim = embedding_dim
        self.use_multiscale = use_multiscale
        self.use_bnneck = use_bnneck
        
        # === Backbone ===
        if backbone == "swinv2-tiny":
            self.backbone = SwinV2Backbone(
                model_name="microsoft/swinv2-tiny-patch4-window8-256",
                pretrained=pretrained,
                extract_stages=multiscale_stages if use_multiscale else (4,),
            )
            backbone_out_dim = self.backbone.hidden_size
        elif backbone in ("densenet121", "densenet169", "densenet201"):
            self.backbone = DenseNetBackbone(
                model_name=backbone,
                pretrained=pretrained,
                extract_stages=multiscale_stages if use_multiscale else (4,),
            )
            backbone_out_dim = self.backbone.hidden_size
        elif backbone in ("resnet50", "resnet101"):
            self.backbone = ResNetBackbone(
                model_name=backbone,
                pretrained=pretrained,
                extract_stages=multiscale_stages if use_multiscale else (4,),
            )
            backbone_out_dim = self.backbone.hidden_size
        elif backbone in ("convnext-tiny", "convnext-small", "convnext-base"):
            self.backbone = ConvNeXtBackbone(
                model_name=backbone,
                pretrained=pretrained,
                extract_stages=multiscale_stages if use_multiscale else (4,),
            )
            backbone_out_dim = self.backbone.hidden_size
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # === Multi-Scale Fusion ===
        if use_multiscale:
            fusion_out_channels = 512  # Fixed intermediate size
            self.fusion = MultiScaleFeatureFusion(
                stage_channels=self.backbone.stage_channels,
                output_channels=fusion_out_channels,
            )
            pool_in_channels = fusion_out_channels
        else:
            self.fusion = None
            pool_in_channels = backbone_out_dim
        
        # === Pooling ===
        self.gem_pool = GeM(p=3.0, learnable=True)
        
        if use_attention_pooling:
            self.attn_pool = AttentionPooling(pool_in_channels)
            # Concatenate GeM and attention pooling
            pool_out_dim = pool_in_channels * 2
        else:
            self.attn_pool = None
            pool_out_dim = pool_in_channels
        
        # === Embedding Head ===
        self.embedding_head = EmbeddingHead(
            in_dim=pool_out_dim,
            embedding_dim=embedding_dim,
            depth=embedding_head_depth,
            dropout=dropout,
            use_residual=use_residual_head,
        )
        
        # === BNNeck ===
        if use_bnneck:
            self.bnneck = nn.BatchNorm1d(embedding_dim)
            nn.init.ones_(self.bnneck.weight)
            nn.init.zeros_(self.bnneck.bias)
        else:
            self.bnneck = None
    
    def forward(
        self,
        x: torch.Tensor,
        return_normalized: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: (B, 3, H, W) input images
            return_normalized: If True, L2-normalize embeddings
        
        Returns:
            embeddings: (B, embedding_dim)
            
        Note on BNNeck:
            - During training (return_normalized=False): returns pre-BNNeck features
            - During inference (return_normalized=True): returns post-BNNeck, normalized features
        """
        # Extract features
        features, final = self.backbone(x)
        
        # Multi-scale fusion or use final features directly
        if self.fusion is not None:
            fused = self.fusion(features)  # (B, C, H, W)
        else:
            # For SwinV2, need to reshape final to spatial
            if hasattr(self.backbone, 'backbone') and hasattr(self.backbone.backbone, 'config'):
                # SwinV2: final is (B, L, C)
                B, L, C = final.shape
                H = W = int(L ** 0.5)
                fused = final.transpose(1, 2).reshape(B, C, H, W)
            else:
                # DenseNet: final is already (B, C, H, W)
                fused = final
        
        # Pooling
        gem_feat = self.gem_pool(fused)  # (B, C)
        
        if self.attn_pool is not None:
            attn_feat = self.attn_pool(fused)  # (B, C)
            pooled = torch.cat([gem_feat, attn_feat], dim=1)  # (B, 2C)
        else:
            pooled = gem_feat
        
        # Embedding head
        embeddings = self.embedding_head(pooled)  # (B, embedding_dim)
        
        # BNNeck logic
        if self.use_bnneck and self.bnneck is not None:
            if return_normalized:
                # Inference: use post-BNNeck, normalized
                embeddings = self.bnneck(embeddings)
                embeddings = F.normalize(embeddings, p=2, dim=1)
            else:
                # Training: return pre-BNNeck for metric loss
                pass
        elif return_normalized:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    def get_embedding_dim(self) -> int:
        return self.embedding_dim


def create_multiscale_model(config) -> MultiScaleReID:
    """
    Factory function to create model from config.
    
    Args:
        config: Config object with model settings (has .model attribute)
    
    Returns:
        MultiScaleReID model
    """
    model_config = config.model if hasattr(config, 'model') else config
    
    # Handle backbone name mapping
    backbone = getattr(model_config, 'backbone', 'swinv2-tiny')
    
    # Legacy support: if using old 'name' field
    if hasattr(model_config, 'name') and not hasattr(model_config, 'backbone'):
        name = model_config.name.lower()
        if 'swinv2' in name and 'tiny' in name:
            backbone = 'swinv2-tiny'
        elif 'densenet121' in name or 'densenet-121' in name:
            backbone = 'densenet121'
        elif 'densenet169' in name or 'densenet-169' in name:
            backbone = 'densenet169'
    
    return MultiScaleReID(
        backbone=backbone,
        embedding_dim=getattr(model_config, 'embedding_dim', 512),
        dropout=getattr(model_config, 'dropout', 0.1),
        pretrained=getattr(model_config, 'pretrained', True),
        use_multiscale=getattr(model_config, 'use_multiscale', True),
        multiscale_stages=tuple(getattr(model_config, 'multiscale_stages', (2, 3, 4))),
        embedding_head_depth=getattr(model_config, 'embedding_head_depth', 3),
        use_residual_head=getattr(model_config, 'use_residual_head', False),
        use_attention_pooling=getattr(model_config, 'use_attention_pooling', False),
        use_bnneck=getattr(model_config, 'use_bnneck', True),
    )

