"""Importing this package registers all available models with src.models.registry."""
from src.models import dncnn, hybrid_restorer, unet  # noqa: F401
from src.models.registry import build_model  # noqa: F401
