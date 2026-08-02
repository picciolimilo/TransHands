"""Gesture Recognition Task Head."""

import torch.nn as nn


class TCNGestureHead(nn.Module):
    """Temporal Convolutional Network (TCN) Head."""
    def __init__(self, input_dim=256, hidden_dim=512, num_layers=3, num_classes=27, dropout=0.5, kernel_size=3):
        super().__init__()

        layers = []
        in_channels = input_dim

        for i in range(num_layers):
            layers.extend([
                nn.Conv1d(in_channels, hidden_dim, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            in_channels = hidden_dim

        self.conv_layers = nn.Sequential(*layers)

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        """
        Args:
            x: (B, T, J, C) or (B, T, C)
        Returns:
            logits: (B, num_classes)
        """
        if x.dim() == 4:  # (B, T, J, C)
            B, T, J, C = x.shape
            x = x.reshape(B, T, J * C)  # (B, T, J*C)

        # Conv1D expects (B, C, T)
        x = x.transpose(1, 2)  # (B, C, T)

        x = self.conv_layers(x)  # (B, hidden, T)
        x = self.pool(x).squeeze(-1)  # (B, hidden)

        logits = self.classifier(x)
        return logits


def create_gesture_head(head_type='tcn', input_dim=256, num_classes=27, **kwargs):
    """Factory for gesture heads (only 'tcn' is supported)."""
    head_type = head_type.lower()

    if head_type == 'tcn':
        return TCNGestureHead(input_dim, num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown head type: {head_type}")
