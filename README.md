# AI-Based Restoration of Degraded Images — KLA AI Hackathon 2026

Restores images degraded by **speckle noise**, **Gaussian blur/noise**, and
a **fixed 2x resolution reduction** (512→256 or 256→128), reconstructing
each image back to its original resolution and signal quality — built to
the exact spec in `Problem_Statement_01_KLA.pdf`.

**→ See `docs/kla_compliance_checklist.md` for a line-by-line mapping of
every requirement, tip, and FAQ in the KLA deck to exactly where it's
addressed in this repo.** That document is the fastest way to verify
nothing was missed.

**→ See `submission/README.md` for the four required submission
deliverables and the exact commands to produce each one.**

## Why this design

KLA's problem statement (slide 4) is explicit that solutions must be
**both accurate and computationally efficient**, scored on SSIM/PSNR/LPIPS
*and* end-to-end inference time on an NVIDIA H100 (slide 7). That rules out
the heaviest SOTA restoration backbones (full Restormer, SwinIR, diffusion)
as primary choices — they win on quality alone but lose on the speed half
of the score. This repo implements `HybridRestorer`: a NAFNet-lite CNN
trunk (cheap, SOTA-competitive) with a single Restormer-style
channel-attention block at the bottleneck (global context, added only
where it's needed most), conditioned on a lightweight no-reference
degradation estimate (speckle/blur strength) via FiLM modulation, with a
fixed 2x super-resolution head matching KLA's exact downsampling spec. See
`docs/model_comparison.md` for the full architecture comparison and
rationale.

The training loss is deliberately built to target KLA's three official
metrics directly (`docs/loss_functions.md`): a Charbonnier term for PSNR,
an MS-SSIM term for SSIM, and an LPIPS term for LPIPS — directly answering
the deck's own tip (slide 18) to "combine perceptual loss (LPIPS) with
pixel-level metrics (SSIM, pSNR)."

## Data format — confirmed directly from real uploaded sample files

This is not assumed from the deck; it's verified from actual files:

- **Files are `.npy`** (raw float32 NumPy arrays), **not PNG/JPG/TIFF**.
- **Folders are named `NoisyLR/` and `GT/`**, matched by identical
  filename (e.g. `NoisyLR/000040.npy` pairs with `GT/000040.npy`).
- **NoisyLR values are unbounded** — a real checked sample had
  range `[-0.0625, 1.4963]`, confirming values go both negative and above
  1.0 (not just above, as the deck's histograms alone suggested — the
  negative excursion specifically confirms an additive noise component on
  top of multiplicative speckle).
- **GT values are exactly bounded to `[0, 1]`** in the samples checked.
- **GT is exactly 2x the height/width of its paired NoisyLR** — confirmed
  directly (128×128 NoisyLR paired with a 256×256 GT candidate), matching
  the deck's fixed-2x-SR spec.
- KLA's real test set (`Test_NoisyLR/NoisyLR/`) contains **only NoisyLR
  files, no GT** — consistent with it being the held-out set to predict.

```
data/kla_train/
├── NoisyLR/
│   ├── 000040.npy
│   ├── 000054.npy
│   └── ...
└── GT/
    ├── 000040.npy
    ├── 000054.npy
    └── ...
```

(If you extract KLA's release and get a nested `train/train/NoisyLR/`
folder, that's just an archive-structure artifact — point
`configs/data/kla_data.yaml -> data_roots` at whichever folder directly
contains `NoisyLR/` and `GT/`.)

## Repository layout

```
semiconductor-restoration/
├── configs/                  # Hydra-style YAML configs (data/model/train)
├── src/
│   ├── data/
│   │   ├── degradation.py     # KLA-spec speckle + Gaussian + fixed-2x-downsample SYNTHETIC pair generator
│   │   │                       # (optional — KLA's own real NoisyLR/GT pairs need no degradation step)
│   │   └── dataset.py         # loads real (or synthetic) NoisyLR/GT .npy pairs, paired-resolution crop/augment
│   ├── models/
│   │   ├── registry.py
│   │   ├── blocks.py          # NAFBlock, channel-attention block, FiLM conditioning
│   │   ├── dncnn.py / unet.py # baselines
│   │   └── hybrid_restorer.py # <- chosen architecture, fixed 2x SR
│   ├── losses/losses.py       # Charbonnier + MS-SSIM + LPIPS (targets KLA's 3 metrics directly)
│   ├── metrics/metrics.py     # SSIM / PSNR / LPIPS (KLA's exact official metrics)
│   ├── engine/{trainer,evaluator}.py
│   └── utils/
├── scripts/
│   ├── train.py               # KLA deliverable #2: training script
│   ├── evaluate.py            # dev-only quality check (SSIM/PSNR/LPIPS on a held-out split of YOUR OWN data)
│   ├── infer.py                # dev-only manual spot-check tool
│   └── export_onnx.py
├── submission/
│   ├── infer_test_set.py      # KLA deliverable #1: the official evaluation script, .npy in -> .npy out (see submission/README.md)
│   └── README.md               # maps all 4 required deliverables to commands
├── docs/
│   ├── kla_compliance_checklist.md   # <- start here: every tip/FAQ traced to an artifact
│   ├── architecture.md
│   ├── literature_review.md
│   ├── model_comparison.md
│   ├── loss_functions.md
│   ├── evaluation_metrics.md
│   └── experiment_log_template.md
└── tests/test_smoke.py
```

## Quickstart

```bash
pip install -r requirements.txt

# 1. Point configs/data/kla_data.yaml -> data_roots at wherever you
#    extracted KLA's released training set (the folder directly
#    containing NoisyLR/ and GT/).

# 2. (Optional) Generate ADDITIONAL synthetic NoisyLR/GT .npy pairs from
#    your own clean source images, matching KLA's exact degradation spec —
#    useful for widening coverage beyond KLA's released training set
#    (see docs/kla_compliance_checklist.md, "generalization" tip). Writes
#    into the same NoisyLR/GT .npy layout as KLA's real data.
python -m src.data.degradation --input_dir data/my_clean_images --output_dir data/synth --num_variants 4
# then add data/synth as a second entry in configs/data/kla_data.yaml -> data_roots

# 3. Train (Hydra config-driven) — this is the KLA "training script" deliverable
python scripts/train.py data=kla_data model=hybrid_restorer train=default

# 4. Check quality on a held-out split of your own training data (dev tool; reports KLA's 3 official metrics)
python scripts/evaluate.py checkpoint=outputs/checkpoints/best.pt

# 5. Run the OFFICIAL submission evaluation script (KLA deliverable #1) — .npy in, .npy out
python submission/infer_test_set.py --test_dir data/Test_NoisyLR/NoisyLR --output_dir outputs/inference

# 6. Freeze the environment (KLA deliverable #4)
pip freeze > submission/environment_freeze.txt
```

## Key spec details this repo takes care to match exactly

- **Data format is `.npy` float32, not images** — confirmed from real
  files, not assumed. `src/data/dataset.py` and
  `submission/infer_test_set.py` both read/write `.npy` directly, with no
  8-bit/16-bit quantization step anywhere (float32 round-trips losslessly).
- **Fixed 2x resolution ratio** (never a variable SR factor) —
  `configs/model/hybrid_restorer.yaml: sr_scale: 2`, confirmed directly
  from a real 128×128 NoisyLR / 256×256 GT pair.
- **NoisyLR intensity can legitimately be negative or exceed the
  ground-truth range** (confirmed: a real sample had range
  `[-0.0625, 1.4963]`) — never clipped during data generation or
  training; clipped only once, at final inference output
  (`submission/infer_test_set.py::_postprocess`). See
  `docs/kla_compliance_checklist.md` for the full reasoning.
- **No defect masks exist in KLA's dataset** — the defect-preservation
  branch from an earlier iteration of this codebase is disabled by default
  (`use_defect_branch: false`) and every loss/metric that would need a
  mask degrades gracefully to a no-op when one isn't provided.
- **OOD generalization** is a scored dimension (slide 6, 12) — see
  `docs/kla_compliance_checklist.md` for the augmentation/synthetic-data
  strategy and the stratified-reporting recommendation.
