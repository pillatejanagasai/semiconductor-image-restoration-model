"""HybridRestorer — the chosen production architecture for this project.

Design (see docs/architecture.md for the full rationale):

    Input ──▶ IQA-conditioned NAFNet-lite encoder (stages of NAFBlock)
              │
              ▼
        Bottleneck: ChannelAttentionBlock (Restormer-style, global context)
              │
              ▼
        NAFNet-lite decoder (mirrors encoder, skip connections)
              │
        ┌─────┼──────────────┐
        ▼     ▼              ▼
   restoration  defect-seg   confidence (log-variance) head
    head        head (shared decoder features)
        │            │
        └── high-frequency residual re-injection, gated by
            sigmoid(defect_logits), added back into restoration
            output (Defect Preservation Module — prevents the
            network from "restoring away" real defects)

Every design choice here traces to a specific rejected alternative
documented in docs/model_comparison.md:
  - NAFBlock trunk chosen over full Restormer/SwinIR: ~5-8x fewer FLOPs at
    comparable accuracy (NAFNet ablations), required for edge deployment.
  - Channel (not spatial) attention at the bottleneck only: linear
    complexity in image size, avoids the O((HW)^2) blowup of full spatial
    self-attention (e.g. plain ViT) at SEM image resolutions (up to
    several thousand pixels per side in raw tool exports).
  - No GAN/adversarial or diffusion components: rejected due to
    hallucination risk in defect-critical metrology (see literature
    review, section 1.2/1.4).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.blocks import ChannelAttentionBlock, FiLMConditioning, NAFBlock
from src.models.registry import register_model

IQA_VECTOR_DIM = 4  # [noise_sigma_est, blur_sigma_est, resolution_factor_est, illum_gradient_est]


class IQAModule(nn.Module):
    """Lightweight no-reference degradation estimator.

    Produces the 4-D degradation vector [noise, blur, resolution deficit,
    illumination gradient] used to condition the restoration trunk via
    FiLM, and is directly interpretable/loggable for fab QA audit trails
    (see docs/architecture.md, "IQA Module" rationale).
    """

    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(64, IQA_VECTOR_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x).flatten(1)
        return torch.sigmoid(self.head(feat))  # normalized [0,1] degradation estimates


class EncoderStage(nn.Module):
    def __init__(self, channels: int, num_blocks: int, dropout: float, cond_dim: int | None):
        super().__init__()
        self.blocks = nn.ModuleList([NAFBlock(channels, dropout=dropout) for _ in range(num_blocks)])
        self.film = FiLMConditioning(cond_dim, channels) if cond_dim else None

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        if self.film is not None and cond is not None:
            x = self.film(x, cond)
        return x


class HybridRestorer(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        base_width: int = 32,
        enc_blocks: list[int] = (2, 2, 4, 2),
        dec_blocks: list[int] = (2, 2, 2, 2),
        bottleneck_attention_heads: int = 4,
        use_defect_branch: bool = True,
        use_confidence_head: bool = True,
        iqa_conditioning: bool = True,
        sr_scale: int = 1,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.use_defect_branch = use_defect_branch
        self.use_confidence_head = use_confidence_head
        self.iqa_conditioning = iqa_conditioning
        self.sr_scale = sr_scale
        w = base_width
        cond_dim = IQA_VECTOR_DIM if iqa_conditioning else None

        self.iqa_module = IQAModule(in_channels) if iqa_conditioning else None

        self.stem = nn.Conv2d(in_channels, w, kernel_size=3, padding=1)

        # Encoder: N stages, EVERY stage downsamples by 2x (never skipped —
        # see the correctness note below). Total spatial reduction through
        # the encoder is therefore exactly 2**len(enc_blocks).
        #
        # CORRECTNESS NOTE: an earlier version of this code only inserted a
        # downsampling conv when a stage's channel width differed from the
        # previous stage's width (`if stage_w != prev_w else nn.Identity()`).
        # Because the first stage's width equals the stem width, that
        # silently skipped the FIRST downsample, decoupling the encoder's
        # downsample count from the decoder's upsample count (the decoder
        # always performs exactly len(enc_blocks) upsamples to fully mirror
        # the encoder). The mismatch caused the decoder to land one extra
        # 2x upsample past the original input resolution, which then got
        # multiplied again by the SR pixel-shuffle head — silently doubling
        # every output's resolution beyond what `sr_scale` requested (and
        # breaking sr_scale=1 entirely with a shape-mismatch crash on the
        # residual add). Downsampling unconditionally at every stage fixes
        # this: encoder downsamples N times, decoder upsamples N times,
        # landing exactly back at the input resolution before the
        # sr_scale-controlled pixel-shuffle head applies the KLA-spec 2x.
        widths = [w * (2 ** i) for i in range(len(enc_blocks))]
        self.enc_stages = nn.ModuleList()
        self.downs = nn.ModuleList()
        prev_w = w
        for stage_w, n_blocks in zip(widths, enc_blocks):
            self.downs.append(nn.Conv2d(prev_w, stage_w, kernel_size=2, stride=2))
            self.enc_stages.append(EncoderStage(stage_w, n_blocks, dropout, cond_dim))
            prev_w = stage_w

        self.bottleneck = ChannelAttentionBlock(prev_w, num_heads=bottleneck_attention_heads)

        # Decoder: mirrors encoder with skip connections.
        self.up_convs = nn.ModuleList()
        self.dec_stages = nn.ModuleList()
        rev_widths = list(reversed(widths))
        for i in range(1, len(rev_widths)):
            in_w, out_w = rev_widths[i - 1], rev_widths[i]
            self.up_convs.append(nn.ConvTranspose2d(in_w, out_w, kernel_size=2, stride=2))
            self.dec_stages.append(EncoderStage(out_w, dec_blocks[i - 1], dropout, cond_dim))
        # Final stage back to stem width.
        self.up_convs.append(nn.ConvTranspose2d(rev_widths[-1], w, kernel_size=2, stride=2))
        self.dec_stages.append(EncoderStage(w, dec_blocks[-1], dropout, cond_dim))

        # Skip-connection fusion convs (concat -> back to stage width).
        self.skip_fuse = nn.ModuleList(
            [nn.Conv2d(out_w * 2, out_w, kernel_size=1) for out_w in list(reversed(widths))[1:]] + [nn.Conv2d(w * 2, w, kernel_size=1)]
        )

        self.restore_head = nn.Conv2d(w, in_channels * (sr_scale ** 2), kernel_size=1)
        if sr_scale > 1:
            self.pixel_shuffle = nn.PixelShuffle(sr_scale)

        if use_defect_branch:
            self.defect_head = nn.Sequential(
                nn.Conv2d(w, w, kernel_size=3, padding=1), nn.GELU(), nn.Conv2d(w, 1, kernel_size=1)
            )
            # Gate controlling how much high-frequency residual is re-injected
            # in defect regions (Defect Preservation Module).
            self.residual_gate_scale = nn.Parameter(torch.tensor(1.0))

        if use_confidence_head:
            self.log_var_head = nn.Conv2d(w, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if self.iqa_conditioning:
            cond = self.iqa_module(x) if cond is None else cond

        feat = self.stem(x)
        skips = []
        for down, stage in zip(self.downs, self.enc_stages):
            feat = down(feat)
            feat = stage(feat, cond)
            skips.append(feat)

        feat = self.bottleneck(feat)

        skips_rev = list(reversed(skips[:-1])) + [None]  # last skip is the bottleneck input itself, skip it
        for i, (up, stage, fuse) in enumerate(zip(self.up_convs, self.dec_stages, self.skip_fuse)):
            feat = up(feat)
            skip = skips_rev[i] if i < len(skips_rev) - 1 else None
            if skip is not None:
                feat = fuse(torch.cat([feat, skip], dim=1))
            feat = stage(feat, cond)

        decoder_features = feat  # width = stem width `w`, used by all three heads

        restore_out = self.restore_head(decoder_features)
        if self.sr_scale > 1:
            restore_out = self.pixel_shuffle(restore_out)
            base = F.interpolate(x, scale_factor=self.sr_scale, mode="bicubic", align_corners=False)
        else:
            base = x
        restored = base + restore_out  # residual formulation: predict correction, not raw pixels

        defect_logits = None
        if self.use_defect_branch:
            defect_logits = self.defect_head(decoder_features)
            # Defect Preservation Module: re-inject original high-frequency
            # residual in regions the segmentation branch flags as likely
            # defects, so the restoration network cannot "smooth away" them
            # even if the main restoration head over-regularizes locally.
            if self.sr_scale == 1:
                high_freq_residual = x - F.avg_pool2d(x, kernel_size=5, stride=1, padding=2)
                defect_prob = torch.sigmoid(defect_logits)
                restored = restored + self.residual_gate_scale * defect_prob * high_freq_residual

        log_var = self.log_var_head(decoder_features) if self.use_confidence_head else None

        return {
            "restored": restored,
            "defect_logits": defect_logits,
            "log_var": log_var,
            "iqa_vector": cond,
        }

    @torch.no_grad()
    def predict_with_uncertainty(self, x: torch.Tensor, mc_samples: int = 8) -> dict[str, torch.Tensor]:
        """MC-Dropout uncertainty estimation for deployment-time confidence maps.

        Runs `mc_samples` stochastic forward passes with dropout kept
        active, returning the mean restoration and the pixel-wise
        standard deviation across samples as an uncertainty map — cheap,
        no retraining required, complements the trained log-variance head.
        """
        self.train()  # keep dropout active
        samples = []
        for _ in range(mc_samples):
            out = self.forward(x)
            samples.append(out["restored"].unsqueeze(0))
        self.eval()
        stacked = torch.cat(samples, dim=0)
        mean = stacked.mean(dim=0)
        std = stacked.std(dim=0)
        return {"restored": mean, "uncertainty": std}


@register_model("hybrid_restorer")
def build_hybrid_restorer(
    in_channels: int = 1,
    base_width: int = 32,
    enc_blocks: list[int] = (2, 2, 4, 2),
    dec_blocks: list[int] = (2, 2, 2, 2),
    bottleneck_attention_heads: int = 4,
    use_defect_branch: bool = True,
    use_confidence_head: bool = True,
    iqa_conditioning: bool = True,
    sr_scale: int = 1,
    dropout: float = 0.05,
    **_ignored,
) -> HybridRestorer:
    return HybridRestorer(
        in_channels=in_channels,
        base_width=base_width,
        enc_blocks=list(enc_blocks),
        dec_blocks=list(dec_blocks),
        bottleneck_attention_heads=bottleneck_attention_heads,
        use_defect_branch=use_defect_branch,
        use_confidence_head=use_confidence_head,
        iqa_conditioning=iqa_conditioning,
        sr_scale=sr_scale,
        dropout=dropout,
    )
