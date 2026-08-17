# Experiment Log

**Rule: never jump directly to the "best" model/config.** Every change
gets an entry below, in order, following:
`Baseline -> Experiment -> Observation -> Improvement -> Conclusion`.
This is both good R&D practice and exactly what hackathon judges will ask
you to defend ("why this config and not another?").

Copy the block below for each new experiment. Keep entries even for
failed experiments — negative results are evidence too.

---

## Experiment Template

### EXP-000: <short name>

**Date:**
**Config:** `configs/model/<file>.yaml`, `configs/train/<file>.yaml`
(commit/diff or full YAML pasted here)

**Baseline being compared against:** (e.g. EXP-000 itself is the DnCNN
baseline; later experiments compare against the best prior experiment)

**Hypothesis:** What do you expect this change to do, and why (cite the
relevant doc: architecture.md / model_comparison.md / loss_functions.md)?

**Results (fill from `outputs/eval_report.json`, KLA's official metrics):**

| Metric | Baseline | This experiment | Delta |
|---|---|---|---|
| PSNR | | | |
| SSIM | | | |
| LPIPS | | | |
| Edge Preservation Ratio (diagnostic only) | | | |
| Params (M) | | | |
| Inference time — mean ms/image (`submission/infer_test_set.py` `timing_report.json`) | | | |
| Inference time — total wall clock, full test set | | | |

**Observation:** What actually happened — including anything surprising
or contradicting the hypothesis.

**Improvement (next step):** What change does this observation motivate?

**Conclusion:** Keep / discard / modify this change, and why.

---

## Suggested experiment sequence for this project

1. **EXP-001 — Baseline: DnCNN**, plain Charbonnier loss only. Establishes
   the classical-DL floor for PSNR/SSIM/LPIPS.
2. **EXP-002 — Baseline: U-Net**, plain Charbonnier + MS-SSIM. Isolates
   the value of skip connections vs. DnCNN.
3. **EXP-003 — HybridRestorer, reconstruction losses only** (Charbonnier +
   MS-SSIM, no LPIPS term, no IQA conditioning). Isolates the value of
   NAFBlock + channel-attention bottleneck alone.
4. **EXP-004 — + LPIPS training loss.** This is the deck's explicit tip
   (slide 18) — measure the LPIPS delta specifically, and check PSNR/SSIM
   didn't regress as a side effect (they can, if the LPIPS weight is too
   high — see `docs/loss_functions.md`).
5. **EXP-005 — + IQA conditioning (FiLM).** Measure performance
   stratified by degradation severity (bin by the sampled speckle-L /
   blur-sigma values from `src/data/degradation.py`) — should close the
   gap specifically on heavily-degraded samples.
6. **EXP-006 — Loss-weight ablation.** Zero each auxiliary loss term
   (edge/frequency/gradient) one at a time to justify the final weights
   in `configs/train/default.yaml`.
7. **EXP-007 — Architecture size sweep for the speed/quality tradeoff.**
   Compare `base_width` in {16, 24, 32, 48} and report the full
   PSNR/SSIM/LPIPS-vs-inference-time curve — this is the experiment that
   directly answers KLA's "faster... preferred when quality is
   comparable" scoring rule (slide 15). Pick the point on the curve where
   quality stops improving meaningfully per extra millisecond.
8. **EXP-008 — In-distribution vs. out-of-distribution stratified eval.**
   Once KLA's test set (with OOD samples, per slide 12) is available, run
   `submission/infer_test_set.py` on both subsets separately and report
   metrics for each — a single pooled number hides exactly the
   generalization gap KLA is scoring for.
9. **EXP-009 (optional, clearly labeled) — Adversarial loss ablation.**
   Add a small adversarial loss weight, report the LPIPS improvement
   alongside any PSNR/SSIM regression, to empirically justify keeping it
   out of the final pipeline (or justify keeping it in, if the tradeoff
   turns out favorable — don't assume the answer, measure it) — ties back
   to `docs/loss_functions.md`, "Why not adversarial loss?".
10. **EXP-010 — Batch size sweep.** `submission/infer_test_set.py` already
    batches same-shape images together (`--batch_size`, default 8) —
    sweep this value (1, 4, 8, 16, 32) against `timing_report.json`'s
    `total_wall_clock` on the full test set to find the throughput-optimal
    setting for the actual H100 benchmarking hardware, since the optimal
    batch size depends on available GPU memory and image resolution.
11. **EXP-011 — `--compile` and `--tta` tradeoff measurement.** Both flags
    are implemented but OFF by default (see `submission/infer_test_set.py`
    docstring for why). Run the full test set four ways — baseline,
    `--compile` only, `--tta` only, both together — and record
    PSNR/SSIM/LPIPS alongside `total_wall_clock` for each. Only enable a
    flag in your final submission command if its measured quality gain
    (if any, for `--compile`, which should be quality-neutral) is worth
    its measured time cost. Do not enable either based on general
    reputation ("TTA usually helps") without measuring it on this exact
    model and test set.
