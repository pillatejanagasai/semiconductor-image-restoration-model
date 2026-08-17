"""DnCNN baseline (Zhang et al., 2017) — residual Gaussian-denoising CNN.

Included as the classical-DL BASELINE every experiment must be compared
against (see docs/experiment_log_template.md). Predicts the noise residual
rather than the clean image directly, which is what made it beat BM3D
originally. Deliberately kept simple/unconditioned — no IQA conditioning,
no defect branch — so its gap vs. HybridRestorer is directly attributable
to the architectural choices documented in docs/model_comparison.md.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.registry import register_model


class DnCNN(nn.Module):
    def __init__(self, in_channels: int = 1, num_layers: int = 17, features: int = 64):
        super().__init__()
        layers = [nn.Conv2d(in_channels, features, kernel_size=3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(num_layers - 2):
            layers += [
                nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=True),
            ]
        layers += [nn.Conv2d(features, in_channels, kernel_size=3, padding=1)]
        self.body = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        noise_residual = self.body(x)
        restored = x - noise_residual
        return {"restored": restored, "defect_logits": None, "log_var": None}


@register_model("dncnn")
def build_dncnn(in_channels: int = 1, num_layers: int = 17, features: int = 64, **_ignored) -> DnCNN:
    return DnCNN(in_channels=in_channels, num_layers=num_layers, features=features)
