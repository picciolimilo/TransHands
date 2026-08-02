"""Loss functions for hand pose estimation."""

import torch
import torch.nn as nn

# ==========================================
# 2D-TO-3D LIFTING LOSSES
# ==========================================

class WeightedMPJPE(nn.Module):
    """Weighted MPJPE per joint position error."""
    
    def __init__(self):
        super().__init__()
        
        joint_weights = torch.ones(21)
        
        # Intermediate joints (PIP, DIP) get 2x weight
        intermediate_indices = [2, 3, 6, 7, 10, 11, 14, 15, 18, 19]
        for idx in intermediate_indices:
            joint_weights[idx] = 2.0
        
        # Fingertips get extra weight
        fingertip_indices = [4, 8, 12, 16, 20]
        for idx in fingertip_indices:
            joint_weights[idx] = 1.5
        
        self.register_buffer('joint_weights', joint_weights)
    
    def forward(self, pred, target):
        """
        Args:
            pred: (B, T, 21, 3)
            target: (B, T, 21, 3)
        """
        errors = torch.norm(pred - target, dim=-1)  # (B, T, 21)
        weighted_errors = errors * self.joint_weights

        return weighted_errors.mean()

class LiftingMaskedModelingLoss(nn.Module):
    """L-MMM:  Lifting Masked Motion Modeling."""
    
    def __init__(self, 
                 spatial_mask_ratio=0.25,   # Mask 25% of joints
                 temporal_mask_ratio=0.20,  # Mask 20% of frames
                 mask_strategy='spatial',   # 'spatial', 'temporal', or 'both'
                 velocity_weight=0.5):      # Weight for velocity regularization
        super().__init__()
        self.spatial_ratio = spatial_mask_ratio
        self.temporal_ratio = temporal_mask_ratio
        self.mask_strategy = mask_strategy
        self.velocity_weight = velocity_weight
        self.mpjpe = WeightedMPJPE()
    
    def create_spatial_mask(self, B, T, J, device):
        """Random per-joint mask."""
        num_masked = int(J * self.spatial_ratio)

        mask_probs = torch.rand(B, T, J, device=device)
        _, mask_indices = mask_probs.topk(num_masked, dim=-1)
        
        mask = torch.zeros(B, T, J, dtype=torch.bool, device=device)
        mask.scatter_(2, mask_indices, True)
        
        # Wrist is the root joint
        mask[: , : , 0] = False
        
        return mask
    
    def create_temporal_mask(self, B, T, J, device):
        """Random per-frame mask, broadcast across all joints."""
        num_masked = int(T * self.temporal_ratio)

        mask_probs = torch.rand(B, T, device=device)
        _, mask_indices = mask_probs.topk(num_masked, dim=-1)

        mask_temporal = torch.zeros(B, T, dtype=torch.bool, device=device)
        mask_temporal.scatter_(1, mask_indices, True)
        
        # Expand (B, T) to (B, T, J)
        mask = mask_temporal.unsqueeze(-1).expand(-1, -1, J)
        
        return mask
    
    def forward(self, model, input_2d, target_3d):
        """
        Args:
            model: TransHands Model
            input_2d:   (B, T, 21, 2) -> Normalized 2D input
            target_3d:  (B, T, 21, 3) -> Normalized 3D target
        """
        B, T, J, _ = input_2d.shape
        device = input_2d.device
        
        # Create mask
        if self.mask_strategy == 'spatial':
            mask = self.create_spatial_mask(B, T, J, device)
        elif self.mask_strategy == 'temporal':
            mask = self.create_temporal_mask(B, T, J, device)
        else:
            mask_spatial = self.create_spatial_mask(B, T, J, device)
            mask_temporal = self.create_temporal_mask(B, T, J, device)
            mask = mask_spatial | mask_temporal
        
        # Corrupt Input (Zero-out masked 2D joints)
        masked_2d = input_2d.clone()
        masked_2d[mask] = 0.0
        
        # Forward pass
        pred_3d = model(masked_2d)

        # Handle Sequence Length Mismatch (e.g., PoseFormer reduces T)
        T_out = pred_3d.shape[1]
        T_target = target_3d.shape[1]
        
        if T_out == 1 and T > 1:
            # If model outputs only center frame
            mid_idx = T // 2
            mask = mask[:, mid_idx:mid_idx+1]
            if T_target > 1:
                target_3d = target_3d[:, mid_idx:mid_idx+1]
        
        # Compute Reconstruction Loss
        errors = torch.norm(pred_3d - target_3d, dim=-1)  # (B, T_out, 21)
        
        # Boost weight for masked parts (Hard Negative Mining concept)
        mask_weight = 1.0 / self.spatial_ratio  # ~3.33 with 30% masking
        weights = torch.ones_like(errors)
        weights[mask] = mask_weight
        
        # Weighted MPJPE on the reconstruction
        loss_pos = (errors * weights * self.mpjpe.joint_weights).mean()
        
        # Velocity Concistency (Temporal Smoothness)
        if self.velocity_weight > 0 and T_out > 1:
            pred_vel = pred_3d[: , 1:] - pred_3d[:, :-1]
            target_vel = target_3d[: , 1:] - target_3d[:, :-1]
            vel_errors = torch.norm(pred_vel - target_vel, dim=-1)
            loss_vel = vel_errors.mean()
            return loss_pos + self.velocity_weight * loss_vel
        
        return loss_pos

# ==========================================
# GESTURE RECOGNITION LOSSES
# ==========================================

class GestureClassificationLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, smoothing=0.1, class_weights=None):
        """
        Args:
            alpha (float): Balancing factor (default: 1.0)
            gamma (float): Focusing parameter. Higher = more focus on hard examples.
                          Typical range: 1.0-3.0 (default: 2.0)
            smoothing (float): Label smoothing amount (0.0-1.0, default: 0.1)
            class_weights (Tensor, optional): Per-class weights for imbalanced data
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing
        
        if class_weights is not None:
            self.register_buffer('class_weights', torch.tensor(class_weights, dtype=torch.float32))
        else:
            self.class_weights = None
    
    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C) predicted class scores for C classes
            targets: (B,) ground truth class indices
        Returns:
            loss: Scalar focal loss value
        """
        num_classes = logits.shape[-1]
        smooth_targets = torch.zeros_like(logits)
        smooth_targets.fill_(self.smoothing / (num_classes - 1))
        smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        ce_loss = -(smooth_targets * log_probs).sum(dim=-1)

        if self.class_weights is not None:
            ce_loss = ce_loss * self.class_weights[targets]

        # Focal loss modulation factor
        probs = torch.exp(log_probs)
        pt = (smooth_targets * probs).sum(dim=-1)  # Probability of correct class
        focal_weight = (1 - pt) ** self.gamma

        loss = self.alpha * focal_weight * ce_loss
        
        return loss.mean()