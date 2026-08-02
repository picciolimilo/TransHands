"""Adapters modules."""

import torch
import torch.nn as nn

try:
    from torchdiffeq import odeint as odeint
    TORCHDIFFEQ_AVAILABLE = True
except ImportError:
    print("Warning: torchdiffeq not installed. Install with: pip install torchdiffeq")
    TORCHDIFFEQ_AVAILABLE = False


class HandMLPAdapter(nn.Module):
    """Simple MLP Topological Adapter."""
    def __init__(self, in_channels, num_hand_joints=21, num_body_joints=17, embed_dim=128):
        super().__init__()
        self.num_body_joints = num_body_joints
        in_dim = num_hand_joints * in_channels
        out_dim = num_body_joints * in_channels

        self.net = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, out_dim)
        )

    def forward(self, x):
        # x: (B, T, 21, C)
        B, T, N_hand, C = x.shape
        x = x.view(B * T, -1)
        x_body = self.net(x)
        return x_body.view(B, T, self.num_body_joints, C)


class ODEFunc(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.Tanh(),
            nn.Linear(dim * 2, dim * 2),
            nn.Tanh(),
            nn.Linear(dim * 2, dim)
        )
        
        # Make dynamics time-aware (optional but helps)
        self.time_encoder = nn.Linear(1, dim)
        
    def forward(self, t, x):
        # t: scalar time
        # x: (B, dim)
        
        # Time encoding (broadcast to batch)
        t_vec = torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype) * t
        t_emb = self.time_encoder(t_vec)
        
        # Dynamics: dx/dt = f(x, t)
        return self.net(x) + t_emb


class HandNeuralODEAdapter(nn.Module):
    """Neural ODE Adapter: Models hand→body transformation as a continuous flow."""
    def __init__(self, in_channels, num_hand_joints=21, num_body_joints=17,
                 embed_dim=128, solver='rk4', use_ode=True):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_body_joints = num_body_joints
        self.use_ode = use_ode and TORCHDIFFEQ_AVAILABLE

        if self.use_ode:
            self.solver = solver
        
        # Hand joints → Latent space
        self.hand_encoder = nn.Sequential(
            nn.Linear(num_hand_joints * in_channels, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        
        # ODE Function or Residual Blocks
        if self.use_ode:
            self.ode_func = ODEFunc(embed_dim)
            # Integration time span [0, 1]
            self.register_buffer('integration_time', torch.tensor([0., 1.]))
        else:
            print("Warning: torchdiffeq not available, using ResNet")
            self.residual_blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(embed_dim, embed_dim),
                    nn.LayerNorm(embed_dim),
                    nn.GELU()
                ) for _ in range(4)
            ])
        
        # Latent space → Body joints
        self.body_decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, num_body_joints * in_channels)
        )
        
    def forward(self, x):
        # x: (B, T, 21, C)
        B, T, N_hand, C = x.shape

        x = x.view(B * T, -1)
        z0 = self.hand_encoder(x)  # (B*T, embed_dim)
        
        if self.use_ode:
            orig_dtype = z0.dtype
            z0_fp32 = z0.to(torch.float32)
            time_fp32 = self.integration_time.to(torch.float32)

            with torch.amp.autocast('cuda', enabled=False):
                z_trajectory = odeint(
                    self.ode_func,
                    z0_fp32,
                    time_fp32,
                    method=self.solver,
                    options={'step_size': 0.25}
                )

            z1 = z_trajectory[-1].to(orig_dtype)
            z1 = torch.clamp(z1, -50.0, 50.0)
        else:
            z1 = z0
            for block in self.residual_blocks:
                z1 = z1 + block(z1)

        x_body = self.body_decoder(z1)  # (B*T, num_body * C)
        x_body = x_body.view(B, T, self.num_body_joints, C)
        
        return x_body

class HandAdapter(nn.Module):
    """ MLP adapter to map encoder embeddings to 21×3 hand joints."""

    def __init__(self, dim=256, hidden=512, num_joints=21):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_joints * 3)
        )
        self.num_joints = num_joints

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Args:
            x: (B, T, dim) - Batch, Time, Embedding Dimension
        Returns:
            out: (B, T, 21, 3) - Predicted 3D coordinates
        """
        B, T, D = x.shape
        x_flat = x.view(B * T, D)
        out = self.net(x_flat)
        return out.view(B, T, self.num_joints, 3)