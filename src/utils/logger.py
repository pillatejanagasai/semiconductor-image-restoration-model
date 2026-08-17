"""Structured logging setup shared across train/eval/infer scripts.

Writes human-readable logs to console + a rotating file under
outputs/<experiment_name>/logs/, so every hackathon run leaves an audit
trail (required for the experiment log in docs/experiment_log_template.md).
"""
import logging
import sys
from pathlib import Path


def get_logger(name: str, log_dir: str | None = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        # Avoid duplicate handlers if get_logger is called multiple times.
        return logger

    logger.setLevel(level)
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    if log_dir is not None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(Path(log_dir) / f"{name}.log")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger
