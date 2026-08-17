"""Reusable building blocks shared across model architectures.

- SimpleGate / NAFBlock: from "Simple Baselines for Image Restoration"
  (NAFNet, ECCV 2022) — activation-free, attention-free block that matches
  or beats heavier transformer restoration models at a fraction of the
  compute. Used as the main trunk of HybridRestorer.
- ChannelAttentionBlock: Restormer-style *channel-wise* (not spatial)
  multi-head self-attention — linear complexity in image resolution,
  used only at the bottleneck for global context.
- FiLMConditioning: feature-wise linear modulation, used to condition the
  restoration trunk on the IQA degradation vector (noise/blur/illum
  estimates) so one network adapts its behavior per-image instead of
  needing per-degradation-type specialist models.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SimpleGate(nn.Module):
    """Splits channels in half and multiplies them — replaces nonlinear
    activations (ReLU/GELU) per the NAFNet ablation showing gating alone
    is sufficient and cheaper."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class SimplifiedChannelAttention(nn.Module):
    """Global-average-pool channel attention without the extra MLP
    nonlinearity NAFNet shows is unnecessary."""

    def __init__(self, channels: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.conv(self.pool(x))


class NAFBlock(nn.Module):
    """Core restoration block: depthwise conv + SimpleGate + simplified
    channel attention, wrapped with LayerNorm and residual connections.
    """

    def __init__(self, channels: int, expansion: int = 2, dropout: float = 0.0):
        super().__init__()
        hidden = channels * expansion

        self.norm1 = nn.GroupNorm(1, channels)  # channel-wise LayerNorm equivalent for conv features
        self.conv1 = nn.Conv2d(channels, hidden, kernel_size=1)
        self.dwconv = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden)
        self.gate1 = SimpleGate()
        self.sca = SimplifiedChannelAttention(hidden // 2)
        self.conv2 = nn.Conv2d(hidden // 2, channels, kernel_size=1)

        self.norm2 = nn.GroupNorm(1, channels)
        self.conv3 = nn.Conv2d(channels, hidden, kernel_size=1)
        self.gate2 = SimpleGate()
        self.conv4 = nn.Conv2d(hidden // 2, channels, kernel_size=1)

        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        # Learnable residual scaling (stabilizes very deep stacks).
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.norm1(x)
        y = self.conv1(y)
        y = self.dwconv(y)
        y = self.gate1(y)
        y = self.sca(y)
        y = self.conv2(y)
        y = self.dropout(y)
        x = residual + y * self.beta

        residual = x
        y = self.norm2(x)
        y = self.conv3(y)
        y = self.gate2(y)
        y = self.conv4(y)
        y = self.dropout(y)
        x = residual + y * self.gamma
        return x


class ChannelAttentionBlock(nn.Module):
    """Restormer-style multi-head channel (not spatial) self-attention.

    Attention is computed over the channel dimension after flattening
    spatial dims, giving O(C^2) complexity instead of O(HW)^2 — tractable
    at full SEM image resolution without tiling. Used only at the
    bottleneck of HybridRestorer where global structural context
    (periodic wafer patterns, long line features) matters most.
    """

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(1, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.qkv_dwconv = nn.Conv2d(channels * 3, channels * 3, kernel_size=3, padding=1, groups=channels * 3, bias=False)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        hidden = channels * 4
        self.ffn_norm = nn.GroupNorm(1, channels)
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden),
            nn.GELU(),
            nn.Conv2d(hidden, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        y = self.norm(x)
        qkv = self.qkv_dwconv(self.qkv(y))
        q, k, v = qkv.chunk(3, dim=1)

        head_dim = c // self.num_heads
        q = q.reshape(b, self.num_heads, head_dim, h * w)
        k = k.reshape(b, self.num_heads, head_dim, h * w)
        v = v.reshape(b, self.num_heads, head_dim, h * w)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = out.reshape(b, c, h, w)
        x = x + self.proj_out(out)

        x = x + self.ffn(self.ffn_norm(x))
        return x


class FiLMConditioning(nn.Module):
    """Feature-wise linear modulation: conditions feature maps on the IQA
    degradation vector (noise sigma, blur sigma, resolution factor,
    illumination gradient) so the SAME network adapts per-image instead
    of requiring specialist models per degradation type.
    """

    def __init__(self, cond_dim: int, channels: int):
        super().__init__()
        self.to_scale_shift = nn.Linear(cond_dim, channels * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale_shift = self.to_scale_shift(cond)
        scale, shift = scale_shift.chunk(2, dim=-1)
        scale = scale.unsqueeze(-1).unsqueeze(-1)
        shift = shift.unsqueeze(-1).unsqueeze(-1)
        return x * (1 + scale) + shift
