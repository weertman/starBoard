"""
Loss functions for metric learning.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class CircleLoss(nn.Module):
    """
    Circle Loss: A Unified Perspective of Pair Similarity Optimization (CVPR 2020)
    
    Key idea: Weight positive and negative similarities differently based on 
    how far they are from the decision boundary.
    """
    
    def __init__(self, margin: float = 0.25, scale: float = 64.0):
        super().__init__()
        self.margin = margin
        self.scale = scale
        self.soft_plus = nn.Softplus()
    
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: (B, D) normalized embeddings
            labels: (B,) identity labels
        
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
        
        # Check if we have any valid positive pairs
        if pos_mask.sum() == 0:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        
        # Circle loss weighting
        pos_sim = sim_matrix * pos_mask
        neg_sim = sim_matrix * neg_mask
        
        # Optimal similarity targets
        delta_p = 1 - self.margin  # Positive target
        delta_n = self.margin       # Negative target
        
        # Detach for computing weights
        alpha_p = torch.clamp(delta_p - pos_sim.detach(), min=0)
        alpha_n = torch.clamp(neg_sim.detach() - delta_n, min=0)
        
        # Compute logits
        logit_p = -self.scale * alpha_p * (pos_sim - delta_p)
        logit_n = self.scale * alpha_n * (neg_sim - delta_n)
        
        # Apply masks
        logit_p = logit_p * pos_mask
        logit_n = logit_n * neg_mask
        
        # Mask out invalid entries (use -1e9 for invalid so they don't affect logsumexp)
        logit_p_masked = torch.where(pos_mask > 0, logit_p, torch.tensor(-1e9, device=logit_p.device))
        logit_n_masked = torch.where(neg_mask > 0, logit_n, torch.tensor(-1e9, device=logit_n.device))
        
        # LogSumExp over pairs
        loss_p = torch.logsumexp(logit_p_masked, dim=1)
        loss_n = torch.logsumexp(logit_n_masked, dim=1)
        
        # Combined loss
        loss = self.soft_plus(loss_p + loss_n).mean()
        
        return loss


class TripletLoss(nn.Module):
    """
    Triplet Loss with hard mining.
    
    For each anchor, finds:
    - Hardest positive: same identity, farthest distance
    - Hardest negative: different identity, closest distance
    """
    
    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin
    
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: (B, D) normalized embeddings
            labels: (B,) identity labels
        
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
        
        # Check for valid pairs
        if pos_mask.sum() == 0:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        
        # Hard positive mining
        pos_dist = dist_matrix * pos_mask
        pos_dist[pos_mask == 0] = -float('inf')
        hardest_pos, _ = pos_dist.max(dim=1)
        
        # Hard negative mining
        neg_dist = dist_matrix * neg_mask
        neg_dist[neg_mask == 0] = float('inf')
        hardest_neg, _ = neg_dist.min(dim=1)
        
        # Triplet loss
        loss = F.relu(hardest_pos - hardest_neg + self.margin)
        
        # Filter to valid anchors
        has_pos = (pos_mask.sum(dim=1) > 0)
        has_neg = (neg_mask.sum(dim=1) > 0)
        valid = has_pos & has_neg
        
        if valid.sum() > 0:
            loss = loss[valid].mean()
        else:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        
        return loss


class CombinedLoss(nn.Module):
    """
    Combined loss for re-identification.
    """
    
    def __init__(
        self,
        circle_weight: float = 0.7,
        triplet_weight: float = 0.3,
        circle_margin: float = 0.25,
        circle_scale: float = 64.0,
        triplet_margin: float = 0.3,
    ):
        super().__init__()
        
        self.circle_loss = CircleLoss(margin=circle_margin, scale=circle_scale)
        self.triplet_loss = TripletLoss(margin=triplet_margin)
        
        self.circle_weight = circle_weight
        self.triplet_weight = triplet_weight
    
    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            embeddings: (B, D) embeddings
            labels: (B,) identity labels
        
        Returns:
            total_loss: Scalar loss
            loss_dict: Dictionary of individual losses
        """
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=embeddings.device)
        
        # Circle loss
        if self.circle_weight > 0:
            circle = self.circle_loss(embeddings, labels)
            loss_dict['circle'] = circle.item()
            total_loss = total_loss + self.circle_weight * circle
        
        # Triplet loss
        if self.triplet_weight > 0:
            triplet = self.triplet_loss(embeddings, labels)
            loss_dict['triplet'] = triplet.item()
            total_loss = total_loss + self.triplet_weight * triplet
        
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict


