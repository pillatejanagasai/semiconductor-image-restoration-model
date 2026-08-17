"""Smoke tests — verify every registered model runs a forward pass at the
KLA-spec fixed 2x SR ratio, the composite loss is finite without any
defect mask present, and metrics compute without error. Run before every
commit: `pytest tests/test_smoke.py -v`.
"""
import torch

from src.losses.losses import CompositeRestorationLoss
from src.metrics.metrics import compute_all_metrics
from src.models import build_model


def _dummy_pair(batch_size=2, channels=1, lr_size=32, sr_scale=2):
    degraded = torch.randn(batch_size, channels, lr_size, lr_size)
    clean = torch.randn(batch_size, channels, lr_size * sr_scale, lr_size * sr_scale)
    return degraded, clean


def test_dncnn_forward_denoise_only():
    # DnCNN has no SR head — used here purely as a denoise-only baseline sanity check.
    model = build_model("dncnn", in_channels=1)
    degraded, _ = _dummy_pair(sr_scale=1)
    out = model(degraded)
    assert out["restored"].shape == degraded.shape


def test_unet_forward_denoise_only():
    model = build_model("unet", in_channels=1, base_width=8, use_defect_branch=False)
    degraded, _ = _dummy_pair(sr_scale=1)
    out = model(degraded)
    assert out["restored"].shape == degraded.shape


def test_hybrid_restorer_forward_matches_kla_2x_sr_spec():
    """This is the model configuration actually used for the KLA submission:
    fixed 2x SR (512->256 or 256->128), no defect branch (no mask data
    exists), confidence head on (self-supervised, no masks needed)."""
    sr_scale = 2
    model = build_model(
        "hybrid_restorer",
        in_channels=1,
        base_width=8,
        enc_blocks=[1, 1, 1, 1],
        dec_blocks=[1, 1, 1, 1],
        bottleneck_attention_heads=2,
        use_defect_branch=False,
        use_confidence_head=True,
        iqa_conditioning=True,
        sr_scale=sr_scale,
        dropout=0.0,
    )
    degraded, clean = _dummy_pair(lr_size=32, sr_scale=sr_scale)
    out = model(degraded)

    assert out["restored"].shape == clean.shape, "restored output must be exactly sr_scale x the input resolution"
    assert out["defect_logits"] is None  # disabled — no mask data in KLA's dataset
    assert out["log_var"].shape == clean.shape
    assert out["iqa_vector"].shape == (2, 4)

    loss_fn = CompositeRestorationLoss(
        {"charbonnier": 1.0, "ms_ssim": 0.2, "lpips": 0.0, "edge": 0.05, "gradient": 0.05,
         "frequency": 0.05, "defect_preservation": 0.0, "segmentation": 0.0, "uncertainty": 0.05}
    )
    losses = loss_fn(out, {"clean": clean})  # no defect_mask key at all — must not be required
    assert torch.isfinite(losses["total"])
    assert "defect_preservation" not in losses  # confirms it's correctly skipped when unweighted/no mask

    metrics = compute_all_metrics(out, {"clean": clean}, compute_lpips=False)
    assert "psnr" in metrics and "ssim" in metrics
    assert "defect_iou_retention" not in metrics  # confirms no mask -> no defect metric attempted


def test_hybrid_restorer_denoise_only_mode_sr_scale_1():
    """Confirms the same architecture also works at sr_scale=1 (pure
    denoising, no upsampling) in case a future KLA test set includes
    same-resolution degraded/clean pairs."""
    model = build_model(
        "hybrid_restorer", in_channels=1, base_width=8, enc_blocks=[1, 1], dec_blocks=[1, 1],
        bottleneck_attention_heads=2, use_defect_branch=False, sr_scale=1, dropout=0.0,
    )
    degraded, clean = _dummy_pair(lr_size=32, sr_scale=1)
    out = model(degraded)
    assert out["restored"].shape == degraded.shape == clean.shape


def test_hybrid_restorer_mc_dropout_uncertainty():
    model = build_model(
        "hybrid_restorer", in_channels=1, base_width=8, enc_blocks=[1, 1], dec_blocks=[1, 1],
        bottleneck_attention_heads=2, use_defect_branch=False, sr_scale=2, dropout=0.1,
    )
    degraded, clean = _dummy_pair(lr_size=16, sr_scale=2)
    out = model.predict_with_uncertainty(degraded, mc_samples=3)
    assert out["restored"].shape == clean.shape
    assert out["uncertainty"].shape == clean.shape
    assert (out["uncertainty"] >= 0).all()
