"""Standalone evaluator: runs a trained checkpoint over a full test split
and reports KLA's three official restoration-quality metrics (SSIM, PSNR,
LPIPS — Problem_Statement_01_KLA.pdf slides 7 & 14), aggregated overall
AND stratified by degradation severity (docs/evaluation_metrics.md,
"Reporting convention") when severity metadata is available (i.e. for
synthetically-generated samples from src/data/degradation.py — see
src/data/dataset.py for how that metadata is loaded).

Inference-TIME benchmarking (the other half of KLA's scoring) is
intentionally NOT done here — that must match KLA's exact measurement
methodology (script startup + I/O + inference + write, on their H100), so
it lives in submission/infer_test_set.py, the actual submission artifact.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from src.metrics.metrics import compute_all_metrics
from src.utils.logger import get_logger

logger = get_logger("evaluator")

# Bucket edges for speckle_L (see configs/data/kla_data.yaml:
# speckle.L_range, default [5, 40]) -- lower L means stronger speckle.
# Update these if you change L_range meaningfully.
_SEVERITY_BUCKETS = [("severe", 0, 15), ("moderate", 15, 27), ("mild", 27, float("inf"))]


def _bucket_for(speckle_L: float) -> str | None:
    if speckle_L is None or math.isnan(speckle_L):
        return None
    for name, lo, hi in _SEVERITY_BUCKETS:
        if lo <= speckle_L < hi:
            return name
    return None


def _aggregate(rows: list[dict], keys: list[str]) -> dict:
    return {k: sum(r[k] for r in rows) / len(rows) for k in keys}


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader, device: str = "cuda", output_path: str | None = None) -> dict:
    model.eval()
    per_sample_results = []

    for batch in loader:
        degraded = batch["degraded"].to(device)
        clean = batch["clean"].to(device)
        ids = batch["id"]
        speckle_L = batch.get("speckle_L")

        outputs = model(degraded)
        metrics = compute_all_metrics(outputs, {"clean": clean}, compute_lpips=True)
        metrics["id"] = ids[0] if isinstance(ids, list) else ids
        metrics["speckle_L"] = float(speckle_L[0]) if speckle_L is not None else float("nan")
        metrics["severity_bucket"] = _bucket_for(metrics["speckle_L"])
        per_sample_results.append(metrics)

    metric_keys = [k for k in per_sample_results[0] if k not in ("id", "speckle_L", "severity_bucket")]
    overall = _aggregate(per_sample_results, metric_keys)

    report = {"overall": overall, "num_samples": len(per_sample_results)}

    # Stratify by severity bucket ONLY if at least some samples have the
    # metadata to support it (synthetically-generated data) — real KLA
    # data with no *.params.json sidecar reports overall metrics only,
    # which is the honest thing to do rather than fabricating buckets.
    bucketed = [r for r in per_sample_results if r["severity_bucket"] is not None]
    if bucketed:
        by_severity = {}
        for name, _, _ in _SEVERITY_BUCKETS:
            rows = [r for r in bucketed if r["severity_bucket"] == name]
            if rows:
                by_severity[name] = {**_aggregate(rows, metric_keys), "num_samples": len(rows)}
        report["by_severity"] = by_severity
    else:
        logger.info(
            "No severity metadata found on any sample (no *.params.json sidecars) — "
            "skipping severity-stratified breakdown, reporting overall metrics only. "
            "This is expected when evaluating on real (non-synthetic) data."
        )

    logger.info(f"Evaluation report (KLA official metrics: ssim, psnr, lpips): {json.dumps(report, indent=2)}")
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump({"report": report, "per_sample": per_sample_results}, f, indent=2)
        logger.info(f"Wrote full evaluation report to {output_path}")

    return report
