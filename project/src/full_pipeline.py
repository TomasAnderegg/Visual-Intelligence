
import torch
import torch.nn as nn
from .encoder_4m import FakeEncoder4M
from .connector import MLPConnector
from .smolvla_wrapper import FakeSmolVLA

class VLAPipeline(nn.Module):
    def __init__(
        self,
        encoder_dim: int = 512,
        smolvla_dim: int = 512,
        num_tokens: int = 196,
        proprio_dim: int = 7,
        action_dim: int = 7,
    ):
        super().__init__()
        self.encoder   = FakeEncoder4M(embed_dim=encoder_dim, num_tokens=num_tokens)
        self.connector = MLPConnector(encoder_dim=encoder_dim, smolvla_dim=smolvla_dim)
        self.smolvla   = FakeSmolVLA(
            token_dim=smolvla_dim,
            proprio_dim=proprio_dim,
            action_dim=action_dim,
            num_tokens=num_tokens,
        )

    def forward(self, inputs: dict, proprio: torch.Tensor) -> torch.Tensor:
        tokens  = self.encoder(inputs)
        tokens  = self.connector(tokens)
        actions = self.smolvla(tokens, proprio)
        return actions