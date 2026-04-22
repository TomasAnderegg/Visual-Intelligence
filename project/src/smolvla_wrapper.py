import torch
import torch.nn as nn

class FakeSmolVLA(nn.Module):
    """
    Fake SmolVLA.
    Input  : tokens (B, T, D) + proprio (B, proprio_dim)
    Output : actions (B, action_dim)
    """
    def __init__(
        self,
        token_dim: int = 512,
        proprio_dim: int = 7,
        action_dim: int = 7,
        num_tokens: int = 196,
    ):
        super().__init__()
        self.action_dim = action_dim

        flat_dim = num_tokens * token_dim + proprio_dim

        self.action_head = nn.Sequential(
            nn.Linear(flat_dim, 256),
            nn.GELU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, tokens: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        B = tokens.shape[0]
        flat = torch.cat([tokens.view(B, -1), proprio], dim=-1)
        actions = self.action_head(flat)
        print(f"[SmolVLA] actions shape: {actions.shape}")
        return actions