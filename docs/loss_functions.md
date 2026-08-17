# Loss Functions

KLA's deck (slide 18) gives an explicit tip: *"Design effective loss
functions: combine perceptual loss (LPIPS) with pixel-level metrics
(SSIM, pSNR)."* The composite loss below is a direct, literal
implementation of that tip — each term is chosen to target one of KLA's
three official evaluation metrics.

| Loss | Concept | Targets which KLA metric | Notes |
|---|---|---|---|
| Charbonnier | mean(sqrt((x-y)²+eps²)) | **PSNR** | Smooth L1 variant, differentiable everywhere (unlike L1 at 0) — standard primary reconstruction loss in modern restoration literature |
| MS-SSIM | Multi-scale structural similarity | **SSIM** | Directly optimizes the same structural-similarity quantity SSIM measures, across multiple scales — relevant since KLA's sample images show multi-scale periodic texture (see the "texture" and "dendrite" sample figures in the deck) |
| **LPIPS** (training-time) | Learned perceptual distance via a frozen pretrained network | **LPIPS** | `TrainingLPIPSLoss` in `src/losses/losses.py` — the literal implementation of the deck's tip. Optional (weight 0 disables it; auto-disables gracefully if the `lpips` package isn't installed) since it's the most compute-expensive term |
| Edge Loss | L1 on Laplacian-filtered images | Indirectly PSNR/SSIM | Penalizes edge blurring directly — relevant since resolution reduction (the deck's third degradation type) is exactly a loss of edge/high-frequency content |
| Gradient Loss | L1 on Sobel-gradient maps | Indirectly PSNR/SSIM | Cheaper complement to edge loss |
| Frequency Loss | L1 on FFT magnitude | Indirectly PSNR/SSIM/LPIPS | Directly targets the high-frequency content lost during the 2x downsampling KLA's degradation applies |
| Adversarial Loss | GAN discriminator loss | Could improve LPIPS, risks PSNR/SSIM | **Not used** — see rationale below |
| Defect Preservation Loss / Segmentation Loss | Mask-weighted L1 / BCE+Dice | N/A | Retained in the codebase for extensibility but **weight 0 by default** — KLA's dataset has no defect masks, so these terms are inert unless a mask is explicitly supplied (see `docs/kla_compliance_checklist.md`) |

## Composite loss (implemented in `src/losses/losses.py::CompositeRestorationLoss`)

Default weights, `configs/train/default.yaml -> loss_weights`:

```
L_total = 1.00 * L_charbonnier      (-> PSNR)
        + 0.20 * L_ms_ssim          (-> SSIM)
        + 0.10 * L_lpips            (-> LPIPS)
        + 0.05 * L_edge
        + 0.05 * L_gradient
        + 0.05 * L_frequency
        + 0.05 * L_uncertainty      (self-supervised confidence head, optional)
```

These are starting points, not final values — run the loss-weight
ablation in `docs/experiment_log_template.md` (EXP-007) before locking
them in, since the relative importance of PSNR vs. SSIM vs. LPIPS in your
final composite score depends on how KLA weights them (not fully
specified in the deck beyond "faster... preferred when quality is
comparable").

## Why not adversarial loss?

GAN-based sharpening can improve LPIPS (it's specifically good at
producing perceptually plausible high-frequency texture) but tends to
*hurt* PSNR/SSIM by synthesizing texture that doesn't pixel-align with
the true signal — and on a scored competition where all three metrics
count, that's a risky trade to make blind. If you want to explore it,
add it as a clearly-labeled ablation (see
`docs/experiment_log_template.md`, EXP-009) reporting all three metrics
side by side rather than assuming it's a free win.
