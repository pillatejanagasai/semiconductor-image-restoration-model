"""Deterministic seeding for reproducible training runs.

Reproducibility is a hard requirement for industrial R&D: every reported
number in docs/experiment_log_template.md must be regenerable from the
recorded seed + config.
"""
import os
import random

import numpy as np
import torch


def set_deterministic_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Seed all RNGs used across the pipeline (Python, NumPy, PyTorch, CUDA).

    Args:
        seed: Global seed value.
        deterministic: If True, forces cuDNN deterministic algorithms.
            This can reduce throughput ~5-15% but is required for
            experiment reproducibility during ablations.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        # Faster, non-deterministic — fine for final large-scale training
        # once the architecture/hyperparameters are locked in.
        torch.backends.cudnn.benchmark = True
