"""RetNet inspired Retention Projection."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiScaleRetention(nn.Module):
    """Multi-scale retention mechanism."""
    def __init__(self, dim, num_heads=4):
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        # Q, K, V projections
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        # Decay rates (one per head, learnable)
        self.gammas = nn.Parameter(torch.ones(num_heads))
        
        # Layer norm (more suitable for (B, T, C) format than GroupNorm)
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        # x: (B, T, dim)
        B, T, C = x.shape

        Q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Vectorized retention mask: (num_heads, T, T)
        positions = torch.arange(T, device=x.device).unsqueeze(0) - \
                    torch.arange(T, device=x.device).unsqueeze(1)  # (T, T)
        positions_clamped = positions.clamp(min=0).float()

        gammas = torch.sigmoid(self.gammas)  # (num_heads,)
        decay = gammas.view(-1, 1, 1) ** positions_clamped.unsqueeze(0)  # (H, T, T)

        causal_mask = torch.tril(torch.ones(T, T, device=x.device))
        retention_mask = decay * causal_mask.unsqueeze(0)  # (H, T, T)

        # Batched attention: (B, H, T, T)
        qk = torch.matmul(Q, K.transpose(-2, -1))
        qk = qk * retention_mask.unsqueeze(0) / math.sqrt(self.head_dim)

        output = torch.matmul(qk, V)  # (B, H, T, head_dim)
        output = output.transpose(1, 2).contiguous().view(B, T, C)

        output = self.norm(output)
        output = self.out_proj(output)

        return output


class RetNetProjection(nn.Module):
    """RetNet-based projection using retention mechanism."""
    def __init__(self, in_dim, out_dim, num_heads=4, num_layers=2):
        super().__init__()
        
        self.norm = nn.LayerNorm(in_dim)
        
        # Input projection
        self.input_proj = nn.Linear(in_dim, out_dim)
        
        # Retention layers
        self.retention_layers = nn.ModuleList([
            MultiScaleRetention(out_dim, num_heads) 
            for _ in range(num_layers)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(out_dim) for _ in range(num_layers)
        ])
        
        # FFN for each layer
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(out_dim, out_dim * 4),
                nn.GELU(),
                nn.Linear(out_dim * 4, out_dim)
            ) for _ in range(num_layers)
        ])
        
    def forward(self, x):
        # x: (B, T, in_dim)
        x = self.norm(x)
        x = self.input_proj(x)

        for retention, norm, ffn in zip(self.retention_layers, self.layer_norms, self.ffns):
            x_ret = retention(norm(x))
            x = x + x_ret
            x = x + ffn(x)

        return x


class LinearProjection(nn.Module):
    """Simple linear projection with LayerNorm."""
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.linear(self.norm(x))
