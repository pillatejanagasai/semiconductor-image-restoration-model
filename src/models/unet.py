"""U-Net baseline (Ronneberger et al., 2015) with optional auxiliary defect
segmentation head sharing the encoder.

Serves two purposes in this project:
  1. A strong, simple BASELINE (see docs/experiment_log_template.md) to
     quantify the gain from NAFBlock/channel-attention in HybridRestorer.
  2. A minimal proof-of-concept for the dual-head (restoration +
     segmentation) pattern reused by HybridRestorer, kept separate here
     so the two architectures can be ablated independently.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.registry import register_model


def conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.GroupNorm(1, out_ch),
        nn.GELU(),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.GroupNorm(1, out_ch),
        nn.GELU(),
    )


class UNetRestorer(nn.Module):
    def __init__(self, in_channels: int = 1, base_width: int = 32, use_defect_branch: bool = True):
        super().__init__()
        w = base_width
        self.enc1 = conv_block(in_channels, w)
        self.enc2 = conv_block(w, w * 2)
        self.enc3 = conv_block(w * 2, w * 4)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = conv_block(w * 4, w * 8)

        self.up3 = nn.ConvTranspose2d(w * 8, w * 4, kernel_size=2, stride=2)
        self.dec3 = conv_block(w * 8, w * 4)
        self.up2 = nn.ConvTranspose2d(w * 4, w * 2, kernel_size=2, stride=2)
        self.dec2 = conv_block(w * 4, w * 2)
        self.up1 = nn.ConvTranspose2d(w * 2, w, kernel_size=2, stride=2)
        self.dec1 = conv_block(w * 2, w)

        self.restore_head = nn.Conv2d(w, in_channels, kernel_size=1)

        self.use_defect_branch = use_defect_branch
        if use_defect_branch:
            self.defect_head = nn.Conv2d(w, 1, kernel_size=1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        restored = x + self.restore_head(d1)  # residual formulation
        defect_logits = self.defect_head(d1) if self.use_defect_branch else None
        return {"restored": restored, "defect_logits": defect_logits, "log_var": None}


@register_model("unet")
def build_unet(in_channels: int = 1, base_width: int = 32, use_defect_branch: bool = True, **_ignored) -> UNetRestorer:
    return UNetRestorer(in_channels=in_channels, base_width=base_width, use_defect_branch=use_defect_branch)
