# System Architecture

## Pipeline

```
                    ┌──────────────────────────────────────────┐
                    │      INPUT: degraded image (256² or 128²)  │
                    │   speckle noise + Gaussian blur/noise +    │
                    │   already downsampled 2x from source        │
                    └───────────────────┬────────────────────────┘
                                         │
                    ┌──────────────────────────────────────────┐
                    │   LIGHTWEIGHT NO-REFERENCE IQA MODULE       │
                    │   (src/models/hybrid_restorer.py::IQAModule)│
                    │   estimates speckle/blur strength from the  │
                    │   image itself -> 4-D degradation vector    │
                    └───────────────────┬────────────────────────┘
                                         │ conditions via FiLM
                    ┌──────────────────────────────────────────┐
                    │        HYBRID RESTORATION BACKBONE          │
                    │  NAFNet-lite CNN encoder/decoder (cheap,     │
                    │  SOTA-competitive) + a single Restormer-     │
                    │  style channel-attention block at the        │
                    │  bottleneck (global context) + a fixed 2x    │
                    │  pixel-shuffle super-resolution head          │
                    └───────────────────┬────────────────────────┘
                                         │
                    ┌──────────────────────────────────────────┐
                    │   OPTIONAL CONFIDENCE HEAD (self-supervised) │
                    │   heteroscedastic log-variance, no defect    │
                    │   masks needed — trained jointly for free     │
                    └───────────────────┬────────────────────────┘
                                         │
                    ┌──────────────────────────────────────────┐
                    │   OUTPUT: restored image at 2x resolution    │
                    │   (512² or 256², matching KLA's GT pairs)     │
                    │   clamped to [0,1] here — the one and only    │
                    │   clipping point in the whole pipeline        │
                    └──────────────────────────────────────────┘
```

## Module rationale

### IQA Module
Estimates degradation strength (speckle/blur) directly from the input
image and FiLM-conditions the restoration trunk on it (see
`FiLMConditioning` in `src/models/blocks.py`). This lets a single trained
model adapt its behavior per-image across KLA's stated range of
degradation severities, rather than needing separate models per severity
level, and gives an inspectable, loggable signal if you want to debug
which images the model is treating as "heavily degraded."

### Hybrid Restoration Backbone — why this specific mix
KLA scores both restoration quality *and* end-to-end inference time on an
H100 (slide 7), with an explicit preference for faster pipelines when
quality is comparable (slide 15). That directly rules out the heaviest
SOTA backbones (full Restormer, SwinIR, diffusion) as the primary choice —
see `docs/model_comparison.md` for the full comparison table. NAFNet's own
ablations show most of Restormer's accuracy gain comes from training
recipe and gated feed-forward design, not from attention itself, so a
NAFNet-lite trunk plus exactly one channel-attention block at the
bottleneck (where global context matters most, and where the feature map
is smallest so the attention cost is cheapest) captures most of the
accuracy at a fraction of the compute.

### Fixed 2x super-resolution head
KLA's degradation is always exactly one of two pairs: 512→256 or
256→128 — never a variable factor (slide 6, 10). The architecture reflects
this directly: `sr_scale=2` is fixed in `configs/model/hybrid_restorer.yaml`,
implemented via a `PixelShuffle` head added on top of a U-Net-style
trunk that otherwise operates at, and returns to, the input's native
resolution (see the correctness note in `src/models/hybrid_restorer.py`
for why the encoder/decoder resolution bookkeeping has to be exact here —
an off-by-one-stage bug in an earlier version of this code silently
doubled every output's resolution beyond what `sr_scale` requested).

### Confidence head
A heteroscedastic log-variance head trained jointly via a Gaussian NLL
loss, giving a per-pixel uncertainty map at zero extra inference cost.
Unlike a defect-preservation mechanism (which would need ground-truth
defect masks KLA's dataset doesn't provide), this is fully self-supervised
against the clean image and needs no extra labels — kept on by default
since it's essentially free.

### Why NOT clip intensity anywhere before the final output?
KLA's deck is explicit and repeated on this point (slides 4, 10): degraded
images can have intensity values that exceed the ground truth's range due
to speckle overshoot, and this is *expected, not a bug to fix in
preprocessing*. Clipping at data-generation or training time would erase
exactly the signal excursions the network needs to learn to correct.
The only point in the entire pipeline where a clamp is applied is the
final inference output (`submission/infer_test_set.py::_postprocess`),
because at that point the result is meant to represent a physically valid
restored image, not an in-flight degraded signal.

## What this architecture deliberately does NOT include

An earlier iteration of this codebase (built for a different, defect-
preservation-focused hackathon framing) included a defect-segmentation
branch and a defect-aware high-frequency re-injection mechanism. **KLA's
actual dataset has no defect masks and the deck never asks for defect
preservation** — see `docs/kla_compliance_checklist.md` for the full
gap analysis. That code path still exists (`use_defect_branch` flag) for
extensibility, but is off by default and plays no role in satisfying any
actual KLA requirement.
