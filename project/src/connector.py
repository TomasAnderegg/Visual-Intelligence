import torch
import torch.nn as nn

class MLPConnector(nn.Module):
    """
    Projette les tokens 4M vers l'espace SmolVLA.
    (B, T, encoder_dim) → (B, T, smolvla_dim)
    """
    def __init__(self, encoder_dim: int = 512, smolvla_dim: int = 512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(encoder_dim, smolvla_dim * 2),
            nn.GELU(),
            nn.Linear(smolvla_dim * 2, smolvla_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.mlp(x)
        print(f"[Connector] output shape: {out.shape}")
        return out