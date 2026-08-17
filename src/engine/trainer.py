"""Training loop: mixed precision, gradient clipping, EMA weights,
cosine-annealing-warm-restarts scheduling, checkpointing (best + rolling),
and TensorBoard logging. Deterministic seeding is handled by
src/utils/seed.py and must be called before Trainer construction.
"""
from __future__ import annotations

import copy
from pathlib import Path

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf

from src.losses.losses import CompositeRestorationLoss
from src.metrics.metrics import compute_all_metrics
from src.utils.logger import get_logger


class EMA:
    """Exponential moving average of model weights — reduces variance of
    the final checkpoint, standard practice for restoration models
    (used in Restormer, NAFNet training recipes)."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for shadow_p, model_p in zip(self.shadow.parameters(), model.parameters()):
            shadow_p.mul_(self.decay).add_(model_p.detach(), alpha=1 - self.decay)


class Trainer:
    def __init__(self, model: torch.nn.Module, cfg, device: str = "cuda"):
        self.model = model.to(device)
        self.device = device
        self.cfg = cfg

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.train.optimizer.lr,
            weight_decay=cfg.train.optimizer.weight_decay,
            betas=tuple(cfg.train.optimizer.betas),
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=cfg.train.scheduler.T_0, eta_min=cfg.train.scheduler.eta_min
        )
        self.scaler = GradScaler(enabled=cfg.train.amp)
        self.loss_fn = CompositeRestorationLoss(cfg.train.loss_weights)

        self.ema = EMA(self.model, cfg.train.ema_decay) if cfg.train.ema else None

        self.ckpt_dir = Path(cfg.train.checkpoint.dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(cfg.train.logging.tensorboard_dir)
        self.logger = get_logger("trainer", log_dir=str(self.ckpt_dir.parent / "logs"))

        self.global_step = 0
        self.best_metric = -float("inf")
        self.epochs_without_improvement = 0

    def train_one_epoch(self, loader, epoch: int) -> dict[str, float]:
        self.model.train()
        running = {}
        for batch in loader:
            degraded = batch["degraded"].to(self.device, non_blocking=True)
            clean = batch["clean"].to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=self.cfg.train.amp):
                outputs = self.model(degraded)
                losses = self.loss_fn(outputs, {"clean": clean})

            self.scaler.scale(losses["total"]).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.ema is not None:
                self.ema.update(self.model)

            for k, v in losses.items():
                running[k] = running.get(k, 0.0) + v.item()

            if self.global_step % self.cfg.train.logging.log_every_steps == 0:
                for k, v in losses.items():
                    self.writer.add_scalar(f"train/{k}", v.item(), self.global_step)
                self.logger.info(f"epoch={epoch} step={self.global_step} total_loss={losses['total'].item():.4f}")
            self.global_step += 1

        self.scheduler.step()
        n_batches = len(loader)
        return {k: v / n_batches for k, v in running.items()}

    @torch.no_grad()
    def validate(self, loader, epoch: int) -> dict[str, float]:
        eval_model = self.ema.shadow if self.ema is not None else self.model
        eval_model.eval()
        agg = {}
        n = 0
        for batch in loader:
            degraded = batch["degraded"].to(self.device)
            clean = batch["clean"].to(self.device)
            outputs = eval_model(degraded)
            metrics = compute_all_metrics(outputs, {"clean": clean}, compute_lpips=False)  # LPIPS skipped during frequent val checks for speed; full LPIPS computed by scripts/evaluate.py
            for k, v in metrics.items():
                agg[k] = agg.get(k, 0.0) + v
            n += 1
        avg = {k: v / max(n, 1) for k, v in agg.items()}
        for k, v in avg.items():
            self.writer.add_scalar(f"val/{k}", v, epoch)
        self.logger.info(f"[val] epoch={epoch} " + " ".join(f"{k}={v:.4f}" for k, v in avg.items()))
        return avg

    def save_checkpoint(self, epoch: int, metrics: dict, is_best: bool) -> None:
        state = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "ema_state": self.ema.shadow.state_dict() if self.ema else None,
            "optimizer_state": self.optimizer.state_dict(),
            "metrics": metrics,
            # Stored as a plain nested dict/list (NOT the raw OmegaConf
            # DictConfig) so that anything loading this checkpoint later —
            # in particular submission/infer_test_set.py, which is
            # required to be a standalone script — does not silently need
            # `omegaconf`/`hydra-core` installed just to unpickle it.
            "config": OmegaConf.to_container(self.cfg, resolve=True),
        }
        torch.save(state, self.ckpt_dir / f"epoch_{epoch:04d}.pt")
        if is_best:
            torch.save(state, self.ckpt_dir / "best.pt")
            self.logger.info(f"New best checkpoint at epoch {epoch}: {metrics}")

        # Keep only the last K rolling checkpoints (best.pt is always kept separately).
        rolling = sorted(self.ckpt_dir.glob("epoch_*.pt"))
        keep_k = self.cfg.train.checkpoint.keep_last_k
        for old_ckpt in rolling[:-keep_k]:
            old_ckpt.unlink()

    def fit(self, train_loader, val_loader) -> None:
        monitor_key = self.cfg.train.checkpoint.metric_for_best
        for epoch in range(self.cfg.train.epochs):
            train_metrics = self.train_one_epoch(train_loader, epoch)
            self.logger.info(f"[train] epoch={epoch} " + " ".join(f"{k}={v:.4f}" for k, v in train_metrics.items()))

            if epoch % self.cfg.train.checkpoint.save_every_epochs == 0:
                val_metrics = self.validate(val_loader, epoch)
                current = val_metrics.get(monitor_key, -float("inf"))
                is_best = current > self.best_metric
                if is_best:
                    self.best_metric = current
                    self.epochs_without_improvement = 0
                else:
                    self.epochs_without_improvement += self.cfg.train.checkpoint.save_every_epochs
                self.save_checkpoint(epoch, val_metrics, is_best)

                if (
                    self.cfg.train.early_stopping.enable
                    and self.epochs_without_improvement >= self.cfg.train.early_stopping.patience_epochs
                ):
                    self.logger.info(f"Early stopping at epoch {epoch} (no improvement in {monitor_key}).")
                    break

        self.writer.close()
