import torch
import torch.nn as nn

class FakeEncoder4M(nn.Module):
    def __init__(self, embed_dim: int = 512, num_tokens: int = 196):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_tokens = num_tokens

        self.rgb_proj     = nn.Linear(3 * 224 * 224, embed_dim)
        self.depth_proj   = nn.Linear(1 * 224 * 224, embed_dim)
        self.seg_proj     = nn.Linear(1 * 224 * 224, embed_dim)
        self.thermal_proj = nn.Linear(1 * 224 * 224, embed_dim)

        self.token_expand = nn.Linear(embed_dim, num_tokens * embed_dim)

    def forward(self, inputs: dict) -> torch.Tensor:
        B = inputs["rgb"].shape[0]

        rgb_feat     = self.rgb_proj(inputs["rgb"].view(B, -1))
        depth_feat   = self.depth_proj(inputs["depth"].view(B, -1))
        seg_feat     = self.seg_proj(inputs["seg"].view(B, -1))
        thermal_feat = self.thermal_proj(inputs["thermal"].view(B, -1))

        fused  = rgb_feat + depth_feat + seg_feat + thermal_feat
        tokens = self.token_expand(fused)
        tokens = tokens.view(B, self.num_tokens, self.embed_dim)

        print(f"[Encoder4M] output shape: {tokens.shape}")
        return tokens