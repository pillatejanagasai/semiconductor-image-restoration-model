#!/usr/bin/env python3
"""KLA AI Hackathon — Official Submission Evaluation Script.

This is deliverable #1 from Problem_Statement_01_KLA.pdf, slide 17:
    "Evaluation Script — standalone Python script (non-notebook).
     Accepts: path to test images directory + path to output directory.
     Loads the trained model and runs inference on all input images.
     Writes denoised outputs to the specified directory.
     Must run without manual edits; used as-is for benchmarking."

DATA FORMAT — confirmed directly from real uploaded sample files, not
assumed from the deck: KLA's test set (``Test_NoisyLR/NoisyLR/``) contains
``.npy`` files (raw float32 NumPy arrays), NOT PNG/JPG images. A real
checked sample was shape (128, 128), dtype float32, with values ranging
from -0.0625 to 1.4963 -- confirming values legitimately go both negative
and above 1.0, not just above as the deck's histograms alone suggested.
This script reads and writes ``.npy`` accordingly (see
src/data/dataset.py for the same format used during training).

USAGE (exactly what KLA's benchmarking harness will run):

    python submission/infer_test_set.py \\
        --test_dir /path/to/Test_NoisyLR/NoisyLR \\
        --output_dir /path/to/write/restored/npy

Optional flags (all have working defaults, so the command above alone is
sufficient — no manual edits needed):

    --checkpoint  path to the trained model weights (default:
                  outputs/checkpoints/best.pt, the file produced by
                  scripts/train.py)
    --device      cuda | cpu (default: auto-detect)
    --amp         use mixed precision on GPU for faster inference (default: on)
    --batch_size  arrays of the same resolution are grouped and processed
                  together for GPU throughput (default: 8; set to 1 to
                  process strictly one array at a time)
    --compile     wrap the model with torch.compile() for faster steady-
                  state inference on supported GPUs (default: off). Off
                  by default because the FIRST batch pays a one-time
                  graph-compilation cost that can make total_wall_clock
                  WORSE on a small test set even though later batches get
                  faster — measure both ways on your actual test-set size
                  before deciding (see docs/experiment_log_template.md,
                  EXP-011) rather than assuming it's a free win.
    --tta         horizontal-flip test-time augmentation: runs each array
                  through the model twice (original + flipped) and
                  averages the two restorations for a small quality boost
                  (default: off). Off by default because it roughly
                  DOUBLES inference time, which works directly against
                  KLA's "faster preferred when quality is comparable"
                  scoring rule (slide 15) — only enable this if you've
                  measured that the SSIM/PSNR/LPIPS gain is worth the
                  time cost for your specific submission (see
                  docs/experiment_log_template.md, EXP-011).
    --preview_png optionally ALSO write an 8-bit PNG preview of each
                  restored array next to the .npy output, purely for
                  quick human visual spot-checks -- the .npy file is
                  always the primary output actually used for scoring
                  (default: off, since it adds write time for no scoring
                  benefit).

Timing methodology matches the KLA deck exactly (slide 15): the reported
"total_wall_clock" in timing_report.json covers script startup and
imports, model initialization, reading every input array from disk,
running inference on the full test set, and writing every output array
back to disk — i.e. everything from process start to process end, not
just the model forward pass.

Design notes relevant to KLA's stated requirements/FAQs:
  - This script imports from the `src` package (the same architecture code
    used for training) rather than being a single monolithic file. This
    satisfies "standalone... runs without manual edits" (slide 17/22) --
    standalone means the command above works as-is with zero interactive
    changes, not that the file must have zero imports. The full `src`
    package is part of the submitted repository.
  - Model architecture hyperparameters are NOT hardcoded here. They are
    read from the checkpoint's saved config (a plain dict -- see
    src/engine/trainer.py, Trainer.save_checkpoint -- so loading a
    checkpoint here does NOT require omegaconf/hydra to be installed,
    despite training using Hydra configs), so this script works
    unmodified for any checkpoint produced by scripts/train.py.
  - Output values ARE clipped to [0, 1] here (and only here) before
    saving -- see the comment at `_postprocess` below for why that is the
    one place clipping is appropriate. This matches real GT samples
    checked, which were exactly bounded to [0, 1].
  - Handles both the 512->256 and 256->128 cases from the KLA spec
    automatically: the model's SR scale is fixed at 2x (see
    configs/model/hybrid_restorer.yaml), so whatever resolution an input
    array arrives at, the output is produced at exactly 2x that
    resolution -- matching either degradation pair KLA describes.
  - Arrays are batched by shape for throughput (KLA deck tip, slide 18:
    "batching... matters") but a test set mixing many different
    resolutions will still fall back toward small/singleton batches per
    shape group -- this optimizes the common case (KLA's test set is
    expected to be predominantly one or two fixed resolutions) without
    requiring padding logic for the general case.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict

_T_SCRIPT_START = time.time()

import numpy as np  # noqa: E402
import torch  # noqa: E402

# Make the repository root importable regardless of the working directory
# the benchmarking harness launches this script from.
from pathlib import Path  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.models import build_model  # noqa: E402

_T_IMPORTS_DONE = time.time()


def _read_array(path: Path) -> np.ndarray:
    """Loads a .npy file as float32. NOT normalized/rescaled -- KLA's
    real NoisyLR arrays are already float32 in their native (unbounded)
    range, confirmed directly from a real sample (range=[-0.0625,
    1.4963]). Rescaling by an assumed factor (e.g. /255) here would be
    WRONG and would corrupt real input signal -- unlike 8-bit images,
    these arrays carry no implicit integer encoding to undo."""
    arr = np.load(path).astype(np.float32)
    if arr.ndim == 3:  # defensive: collapse an accidental channel dim if present
        arr = arr.squeeze()
    if arr.ndim != 2:
        raise ValueError(f"{path}: expected a 2D (H, W) array, got shape {arr.shape}")
    return arr


def _postprocess(restored_model_space: torch.Tensor) -> np.ndarray:
    """Converts a model output (normalized to roughly [-1, 1], see
    src/data/dataset.py) back to a savable float32 array in [0, 1].

    Clipping happens HERE and only here: an in-flight degraded input is
    allowed to exceed [0, 1] because that is real signal the model needs
    to see and correct (confirmed necessary by real data -- see module
    docstring). But the model's OUTPUT is the final restored array meant
    to match KLA's GT convention -- real GT samples checked were exactly
    bounded to [0, 1] -- so clamping at this one final step is the
    correct and only place to do it.
    """
    arr = restored_model_space.squeeze().clamp(-1, 1).cpu().numpy() * 0.5 + 0.5
    return np.clip(arr, 0, 1).astype(np.float32)


def _forward(model: torch.nn.Module, x: torch.Tensor, use_amp: bool) -> torch.Tensor:
    if use_amp:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            return model(x)["restored"]
    return model(x)["restored"]


def _forward_with_optional_tta(model: torch.nn.Module, x: torch.Tensor, use_amp: bool, tta: bool) -> torch.Tensor:
    """Optional horizontal-flip TTA: average the model's prediction on the
    original array with its prediction on the horizontally-flipped array
    (unflipped back before averaging). See the --tta flag docstring above
    for the speed/quality tradeoff this introduces -- off by default."""
    restored = _forward(model, x, use_amp)
    if not tta:
        return restored
    restored_flipped = torch.flip(_forward(model, torch.flip(x, dims=[-1]), use_amp), dims=[-1])
    return (restored + restored_flipped) / 2.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="KLA AI Hackathon submission: run trained restoration model on a test-set directory of .npy arrays."
    )
    parser.add_argument("--test_dir", required=True, help="Directory of degraded test .npy arrays (e.g. Test_NoisyLR/NoisyLR).")
    parser.add_argument("--output_dir", required=True, help="Directory to write restored .npy arrays to.")
    parser.add_argument(
        "--checkpoint", default="outputs/checkpoints/best.pt", help="Path to the trained model checkpoint."
    )
    parser.add_argument("--device", default=None, help="cuda | cpu (default: auto-detect).")
    parser.add_argument("--amp", action="store_true", default=True, help="Use mixed precision on GPU (default: on).")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Arrays of the SAME resolution are grouped and run through the model together "
        "for GPU throughput (per the KLA deck's tip, slide 18: 'batching... matters'). "
        "Arrays of differing resolutions are never batched together. Set to 1 to disable batching.",
    )
    parser.add_argument(
        "--compile", action="store_true", default=False, help="Wrap the model with torch.compile() (default: off)."
    )
    parser.add_argument(
        "--tta",
        action="store_true",
        default=False,
        help="Horizontal-flip test-time augmentation for a small quality boost at ~2x inference cost (default: off).",
    )
    parser.add_argument(
        "--preview_png",
        action="store_true",
        default=False,
        help="Also write an 8-bit PNG preview alongside each .npy output, for visual spot-checks only (default: off).",
    )
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    test_dir = Path(args.test_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Model initialization ----
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_cfg = checkpoint["config"]["model"]
    model = build_model(**model_cfg).to(device)
    weights = checkpoint.get("ema_state") or checkpoint["model_state"]
    model.load_state_dict(weights)
    model.eval()

    if args.compile:
        try:
            model = torch.compile(model, mode="reduce-overhead")
        except Exception as e:  # torch.compile can fail on some platforms/torch versions -- fail soft, not hard
            print(f"WARNING: torch.compile() unavailable/failed ({e}); continuing uncompiled.", file=sys.stderr)

    t_model_ready = time.time()

    # ---- Gather input files ----
    image_paths = sorted(test_dir.glob("*.npy"))
    if not image_paths:
        raise FileNotFoundError(f"No .npy files found in {test_dir}")
    t_listing_done = time.time()

    # ---- Read + group by shape (see module docstring: batching tip) ----
    images_by_shape: dict[tuple[int, int], list[tuple[Path, np.ndarray]]] = defaultdict(list)
    for img_path in image_paths:
        arr = _read_array(img_path)
        images_by_shape[arr.shape].append((img_path, arr))
    t_read_done = time.time()

    # ---- Inference loop: batch same-resolution arrays together ----
    per_image_ms = []
    use_amp = args.amp and device == "cuda"
    for shape, items in images_by_shape.items():
        for batch_start in range(0, len(items), args.batch_size):
            batch_items = items[batch_start : batch_start + args.batch_size]
            t0 = time.time()

            batch_arr = np.stack([arr for _, arr in batch_items], axis=0)  # (B, H, W)
            x = torch.from_numpy(batch_arr).float().unsqueeze(1)  # (B, 1, H, W)
            x = (x - 0.5) / 0.5  # match training normalization (src/data/dataset.py)
            x = x.to(device)

            with torch.no_grad():
                restored = _forward_with_optional_tta(model, x, use_amp, args.tta)

            elapsed_ms = (time.time() - t0) * 1000.0
            per_item_ms = elapsed_ms / len(batch_items)  # amortized per-image cost within this batch

            for i, (img_path, _) in enumerate(batch_items):
                out_arr = _postprocess(restored[i : i + 1])
                np.save(output_dir / f"{img_path.stem}.npy", out_arr)
                if args.preview_png:
                    import cv2  # imported lazily -- only needed for this optional debug path

                    cv2.imwrite(str(output_dir / f"{img_path.stem}_preview.png"), (out_arr * 255).astype(np.uint8))
                per_image_ms.append(per_item_ms)

    t_end = time.time()

    # ---- Timing report (methodology matches KLA deck slide 15 exactly) ----
    report = {
        "num_images": len(image_paths),
        "device": device,
        "amp_enabled": use_amp,
        "compile_enabled": args.compile,
        "tta_enabled": args.tta,
        "batch_size": args.batch_size,
        "checkpoint": str(args.checkpoint),
        "timings_seconds": {
            "script_startup_and_imports": round(_T_IMPORTS_DONE - _T_SCRIPT_START, 4),
            "model_initialization": round(t_model_ready - _T_IMPORTS_DONE, 4),
            "file_listing": round(t_listing_done - t_model_ready, 4),
            "array_reading_and_shape_grouping": round(t_read_done - t_listing_done, 4),
            "inference_and_write_total": round(t_end - t_read_done, 4),
            "total_wall_clock": round(t_end - _T_SCRIPT_START, 4),
        },
        "per_image_ms": {
            "mean": round(float(np.mean(per_image_ms)), 3),
            "median": round(float(np.median(per_image_ms)), 3),
            "min": round(float(np.min(per_image_ms)), 3),
            "max": round(float(np.max(per_image_ms)), 3),
        },
    }
    with open(output_dir / "timing_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"\nRestored {len(image_paths)} arrays -> {output_dir}")


if __name__ == "__main__":
    main()
