# Evaluation Metrics

KLA's official scoring (`Problem_Statement_01_KLA.pdf`, slides 7 & 14) uses
exactly three restoration-quality metrics, plus end-to-end inference time
on an NVIDIA H100 GPU. This repo's metrics module mirrors that exactly.

| Metric | What it measures | KLA-official? |
|---|---|---|
| **PSNR** | Pixel-level fidelity (log MSE) | Yes |
| **SSIM** | Luminance/contrast/structural similarity | Yes |
| **LPIPS** | Learned perceptual distance (deep-feature space) | Yes |
| Edge Preservation Ratio (EPR) | Ratio of edge-energy retained vs. ground truth | No — free diagnostic only, useful for spotting over-smoothing during development |
| Inference time (script startup + model init + I/O + inference + write) | Wall-clock time on the benchmarking GPU (H100) | Yes — the other half of the score, alongside quality |

Implemented in `src/metrics/metrics.py`:
- `psnr`, `ssim`, `edge_preservation_ratio` — pure PyTorch, no extra
  dependencies.
- `LPIPSMetric` / the `compute_lpips` flag on `compute_all_metrics` — thin
  wrapper around the optional `lpips` package; degrades gracefully (PSNR/SSIM
  still returned) if `lpips` isn't installed.
- `compute_all_metrics` — aggregator used by `src/engine/trainer.py`
  (val-time, LPIPS skipped for speed during frequent checks) and
  `src/engine/evaluator.py` (full report, LPIPS included).
- `defect_iou_retention` / `critical_dimension_error` — retained for
  extensibility but **not part of KLA's official scoring** (KLA's dataset
  has no defect masks) — see `docs/kla_compliance_checklist.md`.

Inference-time benchmarking is deliberately NOT done inside
`src/engine/evaluator.py` — it must match KLA's own measurement
methodology exactly (script startup, model init, disk I/O, inference on
the full test set, writing outputs — slide 15), so it lives in the actual
submission artifact, `submission/infer_test_set.py`, which self-reports a
`timing_report.json` broken into those same phases.

## Why `metric_for_best: ssim` and not PSNR?

`configs/train/default.yaml` selects the "best" checkpoint by validation
SSIM rather than PSNR. PSNR is a per-pixel log-MSE score and can be
dominated by large flat background regions, rewarding a model that is
slightly blurry everywhere; SSIM's local structural comparison is more
sensitive to whether fine texture/edge structure was actually recovered —
closer to what a human (or LPIPS) would judge as "restored." Since KLA
scores all three metrics together, either is a reasonable single
checkpoint-selection criterion — SSIM is used here as a bias toward
structural fidelity over raw pixel averaging. This is easy to swap: change
`checkpoint.metric_for_best` and `early_stopping.monitor` in
`configs/train/default.yaml` to `psnr` if you'd rather optimize checkpoint
selection for that instead, and consider running both as an ablation (see
`docs/experiment_log_template.md`).

## Reporting convention

`src/engine/evaluator.py::evaluate` automatically reports metrics
**stratified by degradation severity** (bucketed by the sampled speckle-L
value into severe/moderate/mild — see `_SEVERITY_BUCKETS` in that file)
whenever severity metadata is available, i.e. for synthetically-generated
samples produced by `src/data/degradation.py` (which now writes a
`*.params.json` sidecar per sample specifically so this is possible — see
`src/data/dataset.py` for how it's loaded back). Real data with no such
sidecar (e.g. KLA's own released set, before we know its exact format)
reports overall metrics only — the evaluator detects this automatically
and skips fabricating a stratified breakdown it can't support.

Once KLA's real test set is available, additionally report metrics
**stratified by in-distribution vs. out-of-distribution samples**
separately if that split is identifiable from the released data (slide
12: "Purpose: evaluate both accuracy and robustness") — the evaluator
doesn't yet know how to identify OOD samples automatically since KLA
hasn't specified how they'll be labeled/named in the release; add that
split key to `src/engine/evaluator.py` once the format is known, following
the same pattern used for severity buckets.
