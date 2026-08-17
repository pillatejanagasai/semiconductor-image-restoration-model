# Model Selection & Comparison

| Model | Advantages | Disadvantages | Params (approx) | Inference (512², GPU) | Industrial Suitability |
|---|---|---|---|---|---|
| DnCNN | Simple, fast, strong Gaussian baseline | No blur/SR handling, BN hurts blind generalization | ~0.6M | ~2 ms | Good baseline / edge fallback |
| FFDNet | Explicit noise-map conditioning | Denoise-only | ~0.8M | ~2-3 ms | Good — matches IQA-conditioning philosophy |
| U-Net | Strong general baseline, easy dual-head extension | Underperforms modern attention nets on heavy degradation | 5-30M | ~5-8 ms | Excellent as backbone skeleton |
| RCAN | Channel attention recovers periodic structure well | Very deep (400+ layers original) | ~16M | ~15 ms | Moderate — trim depth for edge |
| ESRGAN | Sharp, perceptually pleasing SR | GAN hallucination risk | ~16M (+disc.) | ~10 ms | **Not suitable** — rejected |
| MIRNet | Multi-scale, strong detail retention | Compute-heavy | ~6M | ~20 ms | Good for offline high-fidelity pass |
| MPRNet | Multi-stage progressive refinement | Higher latency/memory | ~20M | ~25-30 ms | Server-side only |
| SwinIR | Global context, strong on periodic content | Window-attention overhead | 11-28M | ~30 ms | Server-side; too heavy for in-line |
| Restormer | Channel-attention transformer, linear complexity, SOTA-competitive | Heavier than pure CNN | ~26M | ~35-40 ms | Best accuracy for server/cloud |
| NAFNet | Best accuracy/compute ratio currently known | Slightly behind Restormer on hardest cases | ~17M (scalable to 2-4M) | ~10-12 ms (full), ~3 ms (lite) | **Best overall pick** |
| Diffusion | Best perceptual quality | 100ms-sec+ latency, hallucination risk | 30-100M+ | 100ms-sec | Offline/forensic only |
| Plain ViT | Global receptive field from layer 1 | Needs large data, quadratic attention at full res | Varies | High | Only as part of a hybrid |
| **Hybrid CNN+Transformer (chosen)** | CNN stem + attention bottleneck + fixed 2x SR head, tunable for edge/server | More engineering effort, more hyperparameters | ~3-15M (configurable) | ~5-15 ms (edge config) | **Chosen backbone** |

## Decision

`HybridRestorer` (`src/models/hybrid_restorer.py`) = NAFNet-lite trunk
(`NAFBlock`, `src/models/blocks.py`) + a single Restormer-style channel
attention block (`ChannelAttentionBlock`) at the bottleneck + IQA-vector
FiLM conditioning + a fixed 2x pixel-shuffle super-resolution head +
a self-supervised heteroscedastic confidence head. (An optional
defect-segmentation head exists in the code — `use_defect_branch` — but
is off by default: KLA's dataset has no defect masks, see
`docs/kla_compliance_checklist.md`.)

**Why not full Restormer?** Full Restormer is accuracy-optimal but its
per-image latency/memory footprint works against KLA's explicit
"faster... preferred when quality is comparable" scoring rule (deck slide
15). NAFNet's ablations show most of Restormer's gains come from training
strategy and gated feed-forward design, not attention itself —
NAFNet-lite + one transformer block captures most of the benefit at a
fraction of the cost.

**Why not ESRGAN/diffusion despite better perceptual scores in general SR
literature?** Both are generative models that can synthesize plausible
but not physically real texture. That's a bad trade specifically because
KLA scores PSNR and SSIM (pixel/structural fidelity) alongside LPIPS
(perceptual quality) — hallucinated texture that improves LPIPS can
actively hurt PSNR/SSIM if it doesn't pixel-align with the true signal,
so it's not a safe default choice for a three-metric scored submission
(see `docs/loss_functions.md`, "Why not adversarial loss?" for the same
argument applied to the loss function instead of the architecture).

## Baselines implemented for comparison

- `dncnn` (`src/models/dncnn.py`) — classical-DL baseline.
- `unet` (`src/models/unet.py`) — strong simple baseline with optional
  defect branch, used to isolate the gain from NAFBlock/channel attention.
- `hybrid_restorer` — the chosen production architecture.

All three share the same `build_model(name, **kwargs)` registry
(`src/models/registry.py`) so ablations only require changing
`model=<name>` in the Hydra config — no script changes needed.
