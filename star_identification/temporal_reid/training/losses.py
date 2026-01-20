"""
Loss functions for metric learning with support for negative-only identities.

Key concept: negative_only_mask
- A boolean tensor where True indicates samples that should ONLY be used as negatives
- These samples are excluded from being anchors or positives in metric losses
- They are excluded entirely from classification (ArcFace) loss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class CircleLoss(nn.Module):
    """
    Circle Loss: A Unified Perspective of Pair Similarity Optimization (CVPR 2020)
    
    Key idea: Weight positive and negative similarities differently based on 
    how far they are from the decision boundary.
    
    Supports negative_only_mask to exclude certain samples from being anchors.
    """
    
    def __init__(self, margin: float = 0.25, scale: float = 64.0):
        super().__init__()
        self.margin = margin
        self.scale = scale
        self.soft_plus = nn.Softplus()
    
    def forward(
        self, 
        embeddings: torch.Tensor, 
        labels: torch.Tensor,
        negative_only_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            embeddings: (B, D) normalized embeddings
            labels: (B,) identity labels
            negative_only_mask: (B,) bool tensor, True = negative-only sample
        
        Returns:
            Scalar loss
        """
        if embeddings.size(0) < 2:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Compute similarity matrix
        sim_matrix = embeddings @ embeddings.t()  # (B, B)
        
        # Create masks
        labels = labels.view(-1, 1)
        pos_mask = (labels == labels.t()).float()  # Same identity
        neg_mask = (labels != labels.t()).float()  # Different identity
        
        # Remove self-similarity from positive mask
        pos_mask = pos_mask - torch.eye(pos_mask.size(0), device=pos_mask.device)
        
        # If negative_only_mask provided, exclude those samples from being anchors
        # They can still be negatives for other samples
        if negative_only_mask is not None:
            # Samples marked as negative_only cannot form positive pairs as anchors
            # Zero out rows for negative_only samples (they can't be anchors)
            anchor_mask = (~negative_only_mask).float().view(-1, 1)  # (B, 1)
            pos_mask = pos_mask * anchor_mask
            
            # Also, negative_only samples can't be positives for each other
            # (since they can't be anchors, this is already handled)
        
        # Check if we have any valid positive pairs
        if pos_mask.sum() == 0:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        
        # Circle loss weighting
        # Alpha_p: weight for positive pairs (higher when similarity is low)
        # Alpha_n: weight for negative pairs (higher when similarity is high)
        
        pos_sim = sim_matrix * pos_mask
        neg_sim = sim_matrix * neg_mask
        
        # Optimal similarity targets
        delta_p = 1 - self.margin  # Positive target
        delta_n = self.margin       # Negative target
        
        # Detach for computing weights (don't backprop through weights)
        alpha_p = torch.clamp(delta_p - pos_sim.detach(), min=0)
        alpha_n = torch.clamp(neg_sim.detach() - delta_n, min=0)
        
        # Compute logits
        logit_p = -self.scale * alpha_p * (pos_sim - delta_p)
        logit_n = self.scale * alpha_n * (neg_sim - delta_n)
        
        # Apply masks
        logit_p = logit_p * pos_mask
        logit_n = logit_n * neg_mask
        
        # Use logsumexp for numerical stability
        # For each anchor, compute: log(sum_neg exp(logit_n)) + log(sum_pos exp(-logit_p))
        
        # Mask out invalid entries with large negative values
        logit_p_masked = torch.where(pos_mask > 0, -logit_p, torch.tensor(-1e9, device=logit_p.device))
        logit_n_masked = torch.where(neg_mask > 0, logit_n, torch.tensor(-1e9, device=logit_n.device))
        
        # LogSumExp over pairs
        loss_p = torch.logsumexp(logit_p_masked, dim=1)  # (B,)
        loss_n = torch.logsumexp(logit_n_masked, dim=1)  # (B,)
        
        # Combined loss with softplus
        loss = self.soft_plus(loss_p + loss_n)
        
        # Only average over valid anchors (not negative_only)
        if negative_only_mask is not None:
            valid_anchors = ~negative_only_mask
            if valid_anchors.sum() > 0:
                loss = loss[valid_anchors].mean()
            else:
                return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        else:
            loss = loss.mean()
        
        return loss


class TripletLoss(nn.Module):
    """
    Triplet Loss with hard mining.
    
    For each anchor, finds:
    - Hardest positive: same identity, farthest distance
    - Hardest negative: different identity, closest distance
    
    Supports negative_only_mask to exclude certain samples from being anchors.
    """
    
    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin
    
    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        negative_only_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            embeddings: (B, D) normalized embeddings
            labels: (B,) identity labels
            negative_only_mask: (B,) bool tensor, True = negative-only sample
        
        Returns:
            Scalar loss
        """
        if embeddings.size(0) < 2:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        
        # Normalize
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Compute pairwise distances
        dist_matrix = 1 - embeddings @ embeddings.t()  # Cosine distance
        
        # Create masks
        labels = labels.view(-1, 1)
        pos_mask = (labels == labels.t()).float()
        neg_mask = (labels != labels.t()).float()
        
        # Remove self from positive mask
        pos_mask = pos_mask - torch.eye(pos_mask.size(0), device=pos_mask.device)
        
        # Exclude negative_only from being anchors
        if negative_only_mask is not None:
            anchor_mask = (~negative_only_mask).float().view(-1, 1)
            pos_mask = pos_mask * anchor_mask
        
        # Check for valid pairs
        if pos_mask.sum() == 0:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        
        # Hard positive mining: farthest positive for each anchor
        pos_dist = dist_matrix * pos_mask
        pos_dist[pos_mask == 0] = -float('inf')  # Ignore non-positives
        hardest_pos, _ = pos_dist.max(dim=1)  # (B,)
        
        # Hard negative mining: closest negative for each anchor
        neg_dist = dist_matrix * neg_mask
        neg_dist[neg_mask == 0] = float('inf')  # Ignore non-negatives
        hardest_neg, _ = neg_dist.min(dim=1)  # (B,)
        
        # Triplet loss
        loss = F.relu(hardest_pos - hardest_neg + self.margin)
        
        # Only average over valid anchors
        if negative_only_mask is not None:
            valid_anchors = ~negative_only_mask
            # Also need to ensure anchor has both positive and negative
            has_pos = (pos_mask.sum(dim=1) > 0)
            has_neg = (neg_mask.sum(dim=1) > 0)
            valid_anchors = valid_anchors & has_pos & has_neg
            
            if valid_anchors.sum() > 0:
                loss = loss[valid_anchors].mean()
            else:
                return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        else:
            # Still filter to anchors with valid pairs
            has_pos = (pos_mask.sum(dim=1) > 0)
            has_neg = (neg_mask.sum(dim=1) > 0)
            valid = has_pos & has_neg
            if valid.sum() > 0:
                loss = loss[valid].mean()
            else:
                return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        
        return loss


class ArcFaceLoss(nn.Module):
    """
    ArcFace: Additive Angular Margin Loss.
    
    Note: This loss is typically DISABLED when using negative-only identities,
    since negative-only identities shouldn't contribute to classification learning.
    """
    
    def __init__(
        self, 
        num_classes: int, 
        embedding_dim: int, 
        margin: float = 0.5, 
        scale: float = 64.0
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.scale = scale
        
        # Class weight matrix
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_normal_(self.weight)
    
    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        negative_only_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            embeddings: (B, D) embeddings (will be normalized)
            labels: (B,) class labels
            negative_only_mask: (B,) bool tensor, True = exclude from loss
        
        Returns:
            Scalar loss
        """
        # Filter out negative_only samples entirely
        if negative_only_mask is not None:
            valid_mask = ~negative_only_mask
            if valid_mask.sum() == 0:
                return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
            embeddings = embeddings[valid_mask]
            labels = labels[valid_mask]
        
        if embeddings.size(0) == 0:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        
        # Normalize embeddings and weights
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)
        
        # Cosine similarity
        cosine = F.linear(embeddings, weight)  # (B, num_classes)
        
        # Get target cosine
        theta = torch.acos(cosine.clamp(-1 + 1e-7, 1 - 1e-7))
        
        # Add margin to target class
        batch_size = embeddings.size(0)
        target_theta = theta[range(batch_size), labels]
        target_cosine_m = torch.cos(target_theta + self.margin)
        
        # Replace target logits
        logits = cosine.clone()
        logits[range(batch_size), labels] = target_cosine_m
        
        # Scale and compute loss
        logits = logits * self.scale
        
        return F.cross_entropy(logits, labels)


class CombinedLoss(nn.Module):
    """
    Combined loss for re-identification.
    
    Combines Circle Loss, Triplet Loss, and optionally ArcFace Loss.
    Properly handles negative_only samples.
    """
    
    def __init__(self, config, num_classes: int):
        super().__init__()
        
        loss_cfg = config.loss
        
        # Initialize losses
        self.circle_loss = CircleLoss(
            margin=loss_cfg.circle_margin,
            scale=loss_cfg.circle_scale,
        )
        
        self.triplet_loss = TripletLoss(
            margin=loss_cfg.triplet_margin,
        )
        
        # ArcFace is optional (often disabled with negative-only setup)
        self.arcface_loss = None
        if loss_cfg.arcface_weight > 0:
            self.arcface_loss = ArcFaceLoss(
                num_classes=num_classes,
                embedding_dim=config.embedding_dim,
                margin=loss_cfg.arcface_margin,
                scale=loss_cfg.arcface_scale,
            )
        
        # Weights
        self.circle_weight = loss_cfg.circle_weight
        self.triplet_weight = loss_cfg.triplet_weight
        self.arcface_weight = loss_cfg.arcface_weight
    
    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        negative_only_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            embeddings: (B, D) embeddings
            labels: (B,) identity labels
            negative_only_mask: (B,) bool tensor, True = negative-only sample
        
        Returns:
            total_loss: Scalar loss
            loss_dict: Dictionary of individual losses for logging
        """
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=embeddings.device)
        
        # Circle loss
        if self.circle_weight > 0:
            circle = self.circle_loss(embeddings, labels, negative_only_mask)
            loss_dict['circle'] = circle.item()
            total_loss = total_loss + self.circle_weight * circle
        
        # Triplet loss
        if self.triplet_weight > 0:
            triplet = self.triplet_loss(embeddings, labels, negative_only_mask)
            loss_dict['triplet'] = triplet.item()
            total_loss = total_loss + self.triplet_weight * triplet
        
        # ArcFace loss (excludes negative_only entirely)
        if self.arcface_loss is not None and self.arcface_weight > 0:
            arcface = self.arcface_loss(embeddings, labels, negative_only_mask)
            loss_dict['arcface'] = arcface.item()
            total_loss = total_loss + self.arcface_weight * arcface
        
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict



