"""Entry point: python scripts/train.py data=kla_data model=hybrid_restorer train=default

This is the project's TRAINING SCRIPT deliverable per KLA submission
requirements (Problem_Statement_01_KLA.pdf, slide 17, item 2): reproduces
training of the submitted model end-to-end from a single command.
"""
import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.data.dataset import KLARestorationDataset
from src.engine.trainer import Trainer
from src.models import build_model
from src.utils.seed import set_deterministic_seed


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    set_deterministic_seed(cfg.seed, cfg.deterministic)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = KLARestorationDataset(
        data_roots=cfg.data.data_roots,
        split="train",
        split_ratios=cfg.data.split,
        patch_size=cfg.data.patch_size,
        sr_scale=cfg.data.sr_scale,
        aug_cfg=cfg.data.augmentation,
        seed=cfg.seed,
    )
    val_ds = KLARestorationDataset(
        data_roots=cfg.data.data_roots,
        split="val",
        split_ratios=cfg.data.split,
        patch_size=cfg.data.patch_size,
        sr_scale=cfg.data.sr_scale,
        aug_cfg=cfg.data.augmentation,
        seed=cfg.seed,
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg.data.batch_size, shuffle=True, num_workers=cfg.data.num_workers,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=cfg.data.num_workers)

    model_cfg = {k: v for k, v in cfg.model.items() if k != "name"}
    model = build_model(cfg.model.name, **model_cfg)

    trainer = Trainer(model, cfg, device=device)
    trainer.fit(train_loader, val_loader)


if __name__ == "__main__":
    main()
