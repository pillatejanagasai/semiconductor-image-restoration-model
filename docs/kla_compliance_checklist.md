# KLA Compliance Checklist

Every requirement, tip, and FAQ from `Problem_Statement_01_KLA.pdf`,
traced to exactly where it's addressed in this repository. Slide numbers
refer to the KLA deck.

## Data format — confirmed from real files, not assumed from the deck

The deck describes degradation types and resolution pairs, but doesn't
specify the actual file format. This was confirmed directly by loading
real uploaded sample files with NumPy:

| Confirmed fact | Evidence | Where addressed in code |
|---|---|---|
| Files are `.npy` (float32 NumPy arrays), not images | Loaded `000040.npy`: `shape=(128,128), dtype=float32` | `src/data/dataset.py`, `submission/infer_test_set.py` both read/write `.npy` directly — no image codec anywhere in the pipeline |
| Folders are named `NoisyLR/` and `GT/`, matched by filename | Screenshot: `train/train/NoisyLR`, `train/train/GT`; `Test_NoisyLR/NoisyLR` | `configs/data/kla_data.yaml -> data_roots`, `src/data/dataset.py::KLARestorationDataset` |
| NoisyLR values are unbounded — can be negative AND exceed 1.0 | `000040.npy` range = `[-0.0625, 1.4963]` | Never clipped pre-inference anywhere in the pipeline — see the intensity-range rows in Section 1 below, now with confirmed real numbers instead of an inference from the deck's histograms alone |
| GT values are exactly bounded to `[0, 1]` | `000281.npy` (candidate GT) range = `[0.0, 1.0]` exactly | `submission/infer_test_set.py::_postprocess` clamps model output to `[0,1]` to match |
| GT is exactly 2x NoisyLR's height/width | `000281.npy` shape `(256,256)` = 2x `000040.npy` shape `(128,128)` | `sr_scale: 2` in `configs/model/hybrid_restorer.yaml`, unchanged — this was already correct, now independently confirmed |
| Test set (`Test_NoisyLR/NoisyLR/`) has no GT folder | Screenshot shows only a `NoisyLR` subfolder | `submission/infer_test_set.py` never reads a GT file — always was the case, now confirmed correct |

**One thing NOT fully confirmed:** which uploaded file (`000281.npy`,
uploaded twice without an explicit label) is actually `GT` vs. a
differently-sized `NoisyLR` sample. The `[0,1]`-exact bounding and 2x
shape ratio make `GT` the near-certain interpretation, but this is an
inference from the data's statistical properties, not an explicit label —
worth double-checking against a few more labeled samples if anything in
training looks off.

## Section 1 — Challenge at a Glance / Why This Matters (slides 3-4)

| KLA statement | Where addressed |
|---|---|
| "Develop AI solutions to restore signal-degraded images" | `src/models/hybrid_restorer.py` |
| Degradation: speckle noise | `src/data/degradation.py::apply_speckle_noise` — Gamma-distributed multiplicative model, chosen to match the "L" (looks) parameter named directly in the deck's own sample-figure titles |
| Degradation: reduction in spatial resolution | `src/data/degradation.py::apply_resolution_reduction` — fixed 2x area-average downsample |
| Goal: reconstruct to original (or near-original) state, including full spatial resolution | `HybridRestorer` with `sr_scale=2` restores exactly to the paired ground-truth resolution |
| "Histograms of degraded vs. ground truth reveal intensity spread"; "Speckle noise can push pixel values beyond the ground truth range" | **Never clipped during data generation or training** — see `src/data/degradation.py` module docstring and `src/data/dataset.py` module docstring. Clipping happens exactly once, at final inference output (`submission/infer_test_set.py::_postprocess_to_uint16`), with the reasoning documented inline at that function |
| "Recovery of fine spatial detail lost during downsampling" | `edge`, `gradient`, `frequency` loss terms (`src/losses/losses.py`) specifically target high-frequency detail retention |
| "Suppression of speckle noise across diverse image types" | Charbonnier + MS-SSIM + LPIPS composite loss (`configs/train/default.yaml -> loss_weights`) |
| "Broad generalization across multiple image distributions" | See "OOD generalization" row below |
| "Solutions must be both accurate and computationally efficient" | Architecture choice explicitly favors NAFNet-lite over heavier Restormer/SwinIR/diffusion for this reason — see `docs/model_comparison.md`, "Decision" section |

## Section 2 — Detailed Problem Statement (slides 9-13)

| KLA statement | Where addressed |
|---|---|
| Degradation types: speckle noise, "Gaussian noise" (described as sharpness/edge-detail reduction — i.e. blur), spatial resolution reduction (512→256 or 256→128) | `src/data/degradation.py` implements exactly these three, with the Gaussian term implemented as blur (matching the deck's own description of its *effect*) plus an optional small additive component — see the module docstring for the reasoning |
| "Restore spatial resolution to 512×512 or 256×256 pixel ground truth" | `sr_scale=2` fixed in `configs/model/hybrid_restorer.yaml` — never a variable/learned scale factor |
| Training set: paired degraded + ground truth | `src/data/dataset.py::KLARestorationDataset` — loads paired samples, synchronized random crop/flip at the correct 2x resolution ratio |
| "Degraded image intensity range may exceed ground truth range... due to speckle noise addition — this is expected behavior" | See intensity-range row above; additionally, `src/data/dataset.py` normalizes without clipping and the model (GroupNorm, not BatchNorm) has no architectural assumption of bounded input |
| "Diverse data origins: models should generalize across multiple image categories and distributions" | See "OOD generalization" row below |
| Test set: in-distribution + out-of-distribution samples, "evaluates both accuracy and robustness" | See "OOD generalization" row below |

## Section 2 — Evaluation (slides 7, 14-15)

| KLA statement | Where addressed |
|---|---|
| Scored on restoration quality + end-to-end inference time on NVIDIA H100 | `submission/infer_test_set.py` is the artifact actually benchmarked; it self-reports timing in the exact categories KLA measures |
| Quality metrics: SSIM, PSNR, LPIPS | `src/metrics/metrics.py::compute_all_metrics` computes exactly these three (plus a free `edge_preservation_ratio` diagnostic). `scripts/evaluate.py` reports them on a held-out split during development |
| "Participants encouraged to design effective loss functions to enhance restoration quality" | `docs/loss_functions.md` + `src/losses/losses.py::CompositeRestorationLoss` — weights chosen specifically to target SSIM/PSNR/LPIPS (see below) |
| Inference time includes: script startup + model init + reading inputs + inference on full test set + writing outputs | `submission/infer_test_set.py` times exactly these phases, separately, and reports a `total_wall_clock` covering all of them — see the script's own docstring for why this specific breakdown was chosen |
| "Teams should optimize the full inference pipeline: data loading and I/O efficiency, batching and memory transfers, inference execution speed" | `submission/infer_test_set.py` uses `torch.no_grad()`, optional AMP (`torch.autocast`) on GPU, a single shared model instance across the whole test set (no reloading per image), and groups same-resolution images into batches (`--batch_size`, default 8) rather than running strictly one image at a time |
| "Faster, well-optimized pipelines preferred when quality is comparable" | Architecture sizing explicitly documented as a quality/speed tradeoff decision in `docs/model_comparison.md` |

## Section 3 — Submission Requirements (slide 17)

| KLA requirement | Where addressed |
|---|---|
| Evaluation Script — standalone Python script (non-notebook), accepts test-images-dir + output-dir, loads model, runs inference, writes outputs, "must run without manual edits" | `submission/infer_test_set.py` — plain `argparse`, exact `--test_dir`/`--output_dir` contract, zero required manual edits (all other flags have working defaults) |
| Training Script — reproduces training | `scripts/train.py` (see `submission/README.md` for the exact command) |
| Denoised Test Outputs | Produced by running `submission/infer_test_set.py` against KLA's released test set (not generatable until KLA releases it — see `submission/README.md`) |
| Environment Specification — complete `pip freeze` output | `submission/README.md`, "4. Environment specification" — command given; must be regenerated from the actual training/eval environment right before final packaging |

## Section 3 — Tips for Participants (slide 18)

| Tip | Where addressed |
|---|---|
| "Prioritize model generalizability — test set includes unseen distributions" | `src/data/degradation.py` samples degradation parameters (speckle L, blur sigma) from wide ranges rather than fixed values, so the model sees varied severity during training. `docs/experiment_log_template.md` EXP-005 explicitly measures performance stratified by degradation severity |
| "Explore intelligent data augmentation... synthetic data generation to cover additional degradation scenarios" | `src/data/degradation.py::build_synthetic_dataset` is exactly this — generate arbitrarily many additional synthetic pairs beyond KLA's released set, at configurable severity ranges (`configs/data/kla_data.yaml`) |
| "...augmentation that simulates varying noise levels and noise types" | `speckle.L_range` and `gaussian.blur_sigma_range` / `additive_sigma_range` in `configs/data/kla_data.yaml` are sampled per-image, not fixed |
| "Design effective loss functions: combine perceptual loss (LPIPS) with pixel-level metrics (SSIM, pSNR)" | `src/losses/losses.py::CompositeRestorationLoss` — Charbonnier (→PSNR) + MS-SSIM (→SSIM) + `TrainingLPIPSLoss` (→LPIPS), all three weighted in `configs/train/default.yaml -> loss_weights`. This is a direct, literal implementation of this exact tip |
| "Optimize the full inference pipeline, not only model architecture: efficient I/O, batching, and GPU memory management matter" | `submission/infer_test_set.py` groups images by identical (H, W) shape and runs each group through the model as a batch (default `--batch_size 8`), rather than one image per forward pass — see the evaluation-timing row above. **Remaining headroom**: batches are currently built by reading all images into memory upfront before inference starts; for very large test sets this trades memory for simplicity — a streaming/chunked variant would reduce peak memory further if that becomes a constraint |
| "Ensure your evaluation script runs end-to-end without manual intervention" | `submission/infer_test_set.py` requires only `--test_dir`/`--output_dir`; every other parameter (architecture, checkpoint format) is self-contained in the checkpoint file itself |

## Section 4 — FAQs (slides 20-22)

| FAQ | Where addressed |
|---|---|
| "What resolution are the training images?" (512×512/256×256 GT, 256×256/128×128 degraded) | `configs/data/kla_data.yaml: sr_scale: 2` supports both pairs identically — the model doesn't hardcode absolute resolution, only the 2x ratio, so it works on either pair without modification |
| "Why does the degraded image intensity range exceed ground truth?" / "This is expected behavior" | Explicitly never clipped pre-inference — see intensity-range rows above |
| "Can I use external data for training?" / synthetic data explicitly encouraged | `src/data/degradation.py` is precisely this capability, ready to run against any additional clean images you source |
| "When is the test set released?" | Not answerable from this repo — refer to the hackathon schedule as the deck states. `submission/README.md` documents the exact command to run once it is released |
| "How is the final score calculated?" (SSIM/PSNR/LPIPS + H100 inference time, faster preferred when quality comparable) | Reflected directly in this repo's metric/loss choices (see rows above) |
| "What does the inference time measurement include?" | `submission/infer_test_set.py`'s `timing_report.json` breaks out exactly these phases so you can verify your own submission's timing profile before final submission |
| "What GPU is used for benchmarking?" (NVIDIA H100, via the team's own evaluation script) | `submission/infer_test_set.py` includes optional AMP (`torch.autocast`) specifically because H100 gets a large throughput benefit from mixed precision — enabled by default when running on CUDA |
| "Should I optimize beyond just model architecture?" (yes — I/O, batching, memory) | See the batching row above — implemented, not just planned |
| "Can I submit a Jupyter notebook as my evaluation script?" (No — must be standalone, non-notebook) | `submission/infer_test_set.py` is a plain `.py` file runnable via `python submission/infer_test_set.py ...`, no notebook anywhere in the submission path |
| "What does 'pip freeze' environment specification mean?" | `submission/README.md`, item 4 — exact command given |
| "Must the evaluation script accept command-line arguments?" (test dir + output dir) | `submission/infer_test_set.py --test_dir ... --output_dir ...` — exact match |
| "Are there restrictions on model architecture?" (No — any AI-driven approach is valid) | `HybridRestorer` is the chosen architecture, but the model registry (`src/models/registry.py`) makes swapping in a different architecture (`dncnn`, `unet`, or a new one) a one-line config change, not a rewrite |

## Things this repo does NOT invent beyond the KLA spec

An earlier iteration of this codebase was built for a different (defect-
preservation-focused) hackathon framing and included a defect-segmentation
branch, a defect-preservation loss, and a "defect-IoU-retention" metric.
**None of that is part of the actual KLA problem statement** — KLA's
dataset has no defect masks and the deck never mentions defect
preservation as a goal. That code is retained in the codebase (behind
`use_defect_branch: false` and zero-weighted loss terms) purely for
extensibility in case defect masks become available later, but it plays
no role in the default training/inference path and should not be presented
as satisfying any KLA requirement — it doesn't correspond to one.
