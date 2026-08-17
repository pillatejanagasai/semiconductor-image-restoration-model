# Literature Review

## 1. Classical Image Restoration

| Method | Mechanism | Strength | Weakness for SEM/wafer images |
|---|---|---|---|
| Wiener Filtering | Frequency-domain deconvolution, known PSF + noise PSD | Fast, closed-form | Assumes stationary Gaussian noise; SEM noise is signal-dependent (Poisson-Gaussian) |
| Non-Local Means | Averages similar patches | Good flat-region denoising | Blurs high-frequency defect edges — bad for defect preservation |
| BM3D | Block-matching + 3D collaborative filtering | Still SOTA-competitive, no training | Can over-smooth sub-pixel defects; no semantic defect/texture distinction |
| Total Variation Deconvolution | Gradient-sparsity regularized deblur | Better edge preservation than Wiener | Staircase artifacts on curved contours |
| Richardson-Lucy | Iterative ML deconvolution | Good for known blur kernel | Diverges/amplifies noise without regularization; blur kernel rarely known precisely in-fab |
| CLAHE / Homomorphic Filtering | Local contrast/illumination correction | Cheap, real-time | Only fixes illumination, not noise/blur |

**Framing for judges:** classical methods treat restoration as generic
signal recovery. Semiconductor inspection images are not generic —
defects are rare, small, high-frequency, and the "correct" answer is not
always the smoothest one.

## 2. Modern Deep Learning Restoration — chronological summary

- **DnCNN (2017)** — residual learning + batch norm, predicts noise
  residual; first CNN to beat BM3D on blind Gaussian denoising.
- **FFDNet (2018)** — noise-level map as input channel, single model
  handles spatially-variant noise.
- **RCAN (2018)** — channel attention + deep residual-in-residual SR;
  good for periodic structure (metal lines, vias).
- **ESRGAN (2018)** — GAN-based SR with perceptual + adversarial loss;
  produces plausible but potentially hallucinated texture — rejected for
  this project.
- **U-Net (2015→ongoing use)** — encoder-decoder with skip connections;
  workhorse for detail-preserving restoration.
- **MIRNet (2020)** — multi-scale residual blocks without full
  downsampling; strong detail retention.
- **MPRNet (2021)** — multi-stage progressive restoration with supervised
  attention modules between stages.
- **SwinIR (2021)** — Swin Transformer backbone, shifted-window
  self-attention, strong long-range structure recovery.
- **Restormer (2022)** — channel-wise (not spatial) attention transformer,
  linear complexity in image size, one of the strongest general
  restoration backbones.
- **NAFNet (2022)** — shows attention/activation complexity isn't
  necessary; simple gating + LayerNorm achieves SOTA at a fraction of the
  compute. Directly relevant to edge deployment.
- **Diffusion-based restoration (2022–2025)** — best perceptual quality,
  but slow (10-100+ sampling steps even distilled) and shares GAN's
  hallucination risk.
- **Domain-specific SEM/wafer work (2023-2025)** — several papers use
  self-supervised approaches (Noise2Noise, Noise2Void) because paired
  clean/degraded SEM data essentially doesn't exist publicly.

## 3. Comparative Summary

| Category | Best for | Worst for |
|---|---|---|
| Classical | No training data, interpretable, real-time | Fine defect preservation, non-stationary noise |
| CNN (DnCNN/FFDNet/RCAN/MIRNet) | Local noise/blur, fast, edge-deployable | Long-range periodic structure, global illumination |
| GAN (ESRGAN) | Perceptual sharpness | Hallucination — unacceptable for defect-critical metrology |
| Transformer (SwinIR/Restormer) | Global context, structure recovery | Compute/memory cost |
| NAFNet-style | Best accuracy/compute tradeoff currently known | Slightly behind Restormer on hardest cases |
| Diffusion | Best perceptual/generative quality | Speed, hallucination risk, hardest to certify |

## 4. Research Gaps relevant to this challenge

1. Most public restoration benchmarks assume additive Gaussian noise;
   comparatively little work targets multiplicative speckle degradation
   combined with a fixed-ratio resolution loss in the same pipeline (the
   exact combination KLA's problem statement specifies).
2. Loss functions in most published pipelines optimize a single metric
   family (either pixel-fidelity or perceptual) rather than being
   explicitly composed to target several simultaneously-scored metrics —
   directly relevant here since KLA scores PSNR, SSIM, and LPIPS together.
3. Quality-vs-inference-time tradeoffs are usually reported qualitatively
   ("real-time capable") rather than benchmarked end-to-end (script
   startup + I/O + inference + write) on a specific target GPU, which is
   exactly how KLA scores submissions.

## 5. This project's approach

A lightweight hybrid CNN+Transformer restoration backbone
(`HybridRestorer`) with a fixed 2x super-resolution head matching KLA's
exact degradation spec, trained with a composite loss explicitly composed
of terms that each target one of KLA's three scored metrics
(Charbonnier→PSNR, MS-SSIM→SSIM, LPIPS→LPIPS — see
`docs/loss_functions.md`), sized specifically to stay competitive on
inference time as well as quality (see `docs/model_comparison.md`,
"Decision").
