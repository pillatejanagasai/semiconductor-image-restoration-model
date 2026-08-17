"""Entry point: python scripts/evaluate.py checkpoint=outputs/checkpoints/best.pt

Dev-only quality check on a held-out split carved out of your OWN training
data (configs/data -> data_roots), reporting KLA's official SSIM/PSNR/LPIPS
metrics. This is NOT the official KLA submission evaluation script -- that
is submission/infer_test_set.py, which runs against KLA's real (unlabeled)
test set and uses a plain argparse CLI, not Hydra.
"""
import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.data.dataset import KLARestorationDataset
from src.engine.evaluator import evaluate
from src.models import build_model


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = cfg.get("checkpoint", "outputs/checkpoints/best.pt")

    test_ds = KLARestorationDataset(
        data_roots=cfg.data.data_roots,
        split="test",
        split_ratios=cfg.data.split,
        patch_size=cfg.data.patch_size,
        sr_scale=cfg.data.sr_scale,
        aug_cfg=cfg.data.augmentation,
        seed=cfg.seed,
    )
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=cfg.data.num_workers)

    model = build_model(cfg.model.name, **cfg.model).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    # Prefer EMA weights if present — lower-variance final model.
    weights = state.get("ema_state") or state["model_state"]
    model.load_state_dict(weights)

    evaluate(model, test_loader, device=device, output_path="outputs/eval_report.json")


if __name__ == "__main__":
    main()
