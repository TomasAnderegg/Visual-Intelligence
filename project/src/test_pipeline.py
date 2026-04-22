import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from src.pipeline.full_pipeline import VLAPipeline

def make_fake_batch(B: int = 2, device: str = "cpu") -> tuple:
    inputs = {
        "rgb":     torch.randn(B, 3, 224, 224, device=device),
        "depth":   torch.randn(B, 1, 224, 224, device=device),
        "seg":     torch.randn(B, 1, 224, 224, device=device),
        "thermal": torch.randn(B, 1, 224, 224, device=device),
    }
    proprio = torch.randn(B, 7, device=device)
    return inputs, proprio


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*50}")
    print(f"Device : {device}")
    print(f"{'='*50}\n")

    pipeline = VLAPipeline(
        encoder_dim=512,
        smolvla_dim=512,
        num_tokens=196,
        proprio_dim=7,
        action_dim=7,
    ).to(device)

    total_params = sum(p.numel() for p in pipeline.parameters())
    print(f"[Pipeline] total params : {total_params:,}\n")

    inputs, proprio = make_fake_batch(B=2, device=device)

    print("--- Forward pass ---")
    with torch.no_grad():
        actions = pipeline(inputs, proprio)

    print(f"\n[FINAL] actions shape : {actions.shape}")
    print(f"[FINAL] actions sample : {actions[0].tolist()}")
    print(f"\n{'='*50}")
    print("✅ Pipeline OK — tout fonctionne.")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()