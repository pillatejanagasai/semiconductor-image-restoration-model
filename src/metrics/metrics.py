"""Evaluation metrics — see docs/evaluation_metrics.md for definitions.

KLA's official scoring (Problem_Statement_01_KLA.pdf, slides 7 & 14) uses
exactly three restoration-quality metrics — SSIM, PSNR, LPIPS — plus
end-to-end inference time on an NVIDIA H100 GPU (timed separately in
submission/infer_test_set.py, not here). `compute_all_metrics` below
returns exactly those three by default; configs/train/default.yaml sets
`metric_for_best: ssim` to select checkpoints (see docs/evaluation_metrics.md
for why SSIM over PSNR). Defect-IoU-Retention is retained for extensibility
(useful if defect masks are ever available) but is NOT part of the
official KLA scoring and is not computed unless a mask is explicitly
passed in.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    import lpips as lpips_lib

    _LPIPS_AVAILABLE = True
except ImportError:
    _LPIPS_AVAILABLE = False


def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 2.0) -> torch.Tensor:
    """PSNR in dB. Inputs assumed normalized to [-1, 1] (data_range=2.0) as
    produced by src/data/transforms.py Normalize(mean=0.5, std=0.5)."""
    mse = F.mse_loss(pred, target, reduction="none").mean(dim=(1, 2, 3))
    return 10 * torch.log10((data_range ** 2) / (mse + 1e-10))


def ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    from src.losses.losses import SSIMLoss

    return 1.0 - SSIMLoss(window_size=window_size)(pred, target)


class LPIPSMetric:
    """Thin wrapper around the `lpips` package (learned perceptual
    similarity). Falls back gracefully with a clear error if the optional
    dependency isn't installed, since it requires downloading pretrained
    VGG/AlexNet weights."""

    def __init__(self, net: str = "alex", device: str = "cuda"):
        if not _LPIPS_AVAILABLE:
            raise ImportError("pip install lpips to use LPIPSMetric")
        self.model = lpips_lib.LPIPS(net=net).to(device)
        self.model.eval()

    @torch.no_grad()
    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # lpips expects 3-channel input; replicate grayscale if needed.
        if pred.shape[1] == 1:
            pred = pred.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        return self.model(pred, target).flatten()


def edge_preservation_ratio(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Ratio of edge-energy retained in the restored image vs. ground
    truth, via Sobel-gradient magnitude. 1.0 = perfectly preserved edge
    content; <1.0 = over-smoothing; >1.0 = over-sharpening/artifacts."""
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
    sobel_y = sobel_x.transpose(2, 3)
    c = pred.shape[1]
    sx, sy = sobel_x.expand(c, 1, 3, 3), sobel_y.expand(c, 1, 3, 3)

    def edge_energy(img):
        gx = F.conv2d(img, sx, padding=1, groups=c)
        gy = F.conv2d(img, sy, padding=1, groups=c)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-8).sum(dim=(1, 2, 3))

    return edge_energy(pred) / (edge_energy(target) + 1e-8)


def defect_iou_retention(
    restored_defect_probs: torch.Tensor, gt_defect_mask: torch.Tensor, threshold: float = 0.5
) -> torch.Tensor:
    """Headline industrial metric: IoU between the defect-segmentation
    branch's prediction on the RESTORED image and the ground-truth defect
    mask. Directly measures whether defects survive restoration, unlike
    PSNR/SSIM which average over the whole (mostly background) image.
    """
    pred_mask = (restored_defect_probs > threshold).float()
    intersection = (pred_mask * gt_defect_mask).sum(dim=(1, 2, 3))
    union = ((pred_mask + gt_defect_mask) > 0).float().sum(dim=(1, 2, 3))
    return intersection / (union + 1e-8)


def critical_dimension_error(pred_edges_px: torch.Tensor, target_edges_px: torch.Tensor, nm_per_px: float) -> torch.Tensor:
    """Converts pixel-level edge-location differences to nanometers, using
    the tool's calibrated pixel pitch — the metric fab metrology teams
    actually report (ΔCD)."""
    return torch.abs(pred_edges_px - target_edges_px) * nm_per_px


_lpips_singleton: "LPIPSMetric | None" = None


def _get_lpips(device: str) -> "LPIPSMetric | None":
    """Lazily construct a single shared LPIPS instance (it loads pretrained
    weights, so we don't want to reconstruct it every call)."""
    global _lpips_singleton
    if not _LPIPS_AVAILABLE:
        return None
    if _lpips_singleton is None:
        _lpips_singleton = LPIPSMetric(net="alex", device=device)
    return _lpips_singleton


@torch.no_grad()
def compute_all_metrics(outputs: dict, targets: dict, compute_lpips: bool = True) -> dict[str, float]:
    """Convenience aggregator used by src/engine/evaluator.py and
    src/engine/trainer.py — computes KLA's three official metrics
    (PSNR, SSIM, LPIPS) plus edge-preservation-ratio as a free diagnostic.
    LPIPS is skipped automatically (with `psnr`/`ssim` still returned) if
    the optional `lpips` package isn't installed.
    """
    pred, clean = outputs["restored"], targets["clean"]
    metrics = {
        "psnr": psnr(pred, clean).mean().item(),
        "ssim": ssim(pred, clean).item(),
        "edge_preservation_ratio": edge_preservation_ratio(pred, clean).mean().item(),
    }
    if compute_lpips:
        lpips_metric = _get_lpips(str(pred.device))
        if lpips_metric is not None:
            metrics["lpips"] = lpips_metric(pred, clean).mean().item()

    mask = targets.get("defect_mask")
    if mask is not None and outputs.get("defect_logits") is not None:
        defect_probs = torch.sigmoid(outputs["defect_logits"])
        metrics["defect_iou_retention"] = defect_iou_retention(defect_probs, mask).mean().item()
    return metrics
