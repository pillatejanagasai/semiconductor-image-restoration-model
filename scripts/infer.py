"""Entry point (developer convenience tool, Hydra-based):
python scripts/infer.py checkpoint=outputs/checkpoints/best.pt input=path/to/000040.npy_or_dir output=outputs/inference

NOTE: this is NOT the official KLA submission deliverable — that is
submission/infer_test_set.py, which uses a fixed plain-argparse CLI
contract (--test_dir/--output_dir), reads/writes .npy exclusively, and
self-times the full pipeline the way KLA's benchmarking harness does.
Use THIS script for quick manual spot-checks during development — unlike
the submission script, it can also dump uncertainty maps and defect masks
(if the optional defect branch is enabled) as PNG visualizations for easy
human viewing, since those are debugging aids, not scored outputs.
"""
from pathlib import Path

import cv2
import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from src.models import build_model
from src.utils.logger import get_logger

logger = get_logger("infer")


def _load_and_preprocess(path: Path, device: str) -> torch.Tensor:
    """Loads a .npy array as-is (float32, NOT rescaled -- KLA's real
    NoisyLR arrays are already in their native unbounded range, see
    src/data/dataset.py module docstring) and normalizes to model space."""
    arr = np.load(path).astype(np.float32)
    if arr.ndim == 3:
        arr = arr.squeeze()
    tensor = torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0)
    tensor = (tensor - 0.5) / 0.5  # match training normalization
    return tensor.to(device)


def _save_npy(tensor: torch.Tensor, path: Path, clip01: bool = True) -> None:
    """Saves a model-space tensor back to [0,1] float32 .npy (matching
    KLA's real GT format)."""
    arr = tensor.squeeze().clamp(-1, 1).cpu().numpy() * 0.5 + 0.5
    if clip01:
        arr = np.clip(arr, 0, 1)
    np.save(path, arr.astype(np.float32))


def _save_png_preview(tensor_01: np.ndarray, path: Path) -> None:
    """8-bit PNG for quick human viewing only -- never the scored output."""
    cv2.imwrite(str(path), (np.clip(tensor_01, 0, 1) * 255).astype(np.uint8))


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = cfg.get("checkpoint", "outputs/checkpoints/best.pt")
    input_path = Path(cfg["input"])
    output_dir = Path(cfg.get("output", "outputs/inference"))
    output_dir.mkdir(parents=True, exist_ok=True)
    use_mc_dropout = cfg.get("uncertainty", False)
    save_png_preview = cfg.get("png_preview", True)  # dev tool default: on, for easy viewing

    model = build_model(cfg.model.name, **cfg.model).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    weights = state.get("ema_state") or state["model_state"]
    model.load_state_dict(weights)
    model.eval()

    image_paths = [input_path] if input_path.is_file() else sorted(input_path.glob("*.npy"))
    if not image_paths:
        raise FileNotFoundError(f"No .npy files found at {input_path}")

    for img_path in image_paths:
        x = _load_and_preprocess(img_path, device)

        if use_mc_dropout:
            out = model.predict_with_uncertainty(x, mc_samples=8)
            restored, uncertainty = out["restored"], out["uncertainty"]
            uncertainty_norm = (uncertainty / uncertainty.max()).squeeze().cpu().numpy()
            if save_png_preview:
                _save_png_preview(uncertainty_norm, output_dir / f"{img_path.stem}_uncertainty.png")
            np.save(output_dir / f"{img_path.stem}_uncertainty.npy", uncertainty_norm.astype(np.float32))
        else:
            with torch.no_grad():
                out = model(x)
            restored = out["restored"]
            if out.get("defect_logits") is not None and save_png_preview:
                defect_prob = torch.sigmoid(out["defect_logits"]).squeeze().cpu().numpy()
                _save_png_preview(defect_prob, output_dir / f"{img_path.stem}_defect_mask.png")

        _save_npy(restored, output_dir / f"{img_path.stem}_restored.npy")
        if save_png_preview:
            restored_01 = (restored.squeeze().clamp(-1, 1).cpu().numpy() * 0.5 + 0.5)
            _save_png_preview(restored_01, output_dir / f"{img_path.stem}_restored_preview.png")

        logger.info(f"Restored {img_path} -> {output_dir / (img_path.stem + '_restored.npy')}")


if __name__ == "__main__":
    main()
