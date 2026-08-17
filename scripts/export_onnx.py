"""Entry point: python scripts/export_onnx.py checkpoint=outputs/checkpoints/best.pt out=model.onnx

Exports the restoration model for edge deployment via ONNX Runtime or
TensorRT (see README.md "Deployment targets" — use base_width=24,
enc_blocks/dec_blocks=[2,2,2,2] config for the edge variant BEFORE
training, then export that checkpoint here).
"""
import hydra
import torch
from omegaconf import DictConfig

from src.models import build_model
from src.utils.logger import get_logger

logger = get_logger("export_onnx")


class ONNXExportWrapper(torch.nn.Module):
    """Wraps the model to return only the restored image tensor — ONNX
    export requires a fixed, simple output signature; defect/uncertainty
    heads can be exported as a second graph if needed for edge QA tooling.
    """

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)["restored"]


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    checkpoint_path = cfg.get("checkpoint", "outputs/checkpoints/best.pt")
    out_path = cfg.get("out", "model.onnx")
    opset = cfg.get("opset", 17)

    model = build_model(cfg.model.name, **cfg.model)
    state = torch.load(checkpoint_path, map_location="cpu")
    weights = state.get("ema_state") or state["model_state"]
    model.load_state_dict(weights)
    model.eval()

    wrapped = ONNXExportWrapper(model)
    dummy_input = torch.randn(1, cfg.model.in_channels, cfg.data.patch_size, cfg.data.patch_size)

    torch.onnx.export(
        wrapped,
        dummy_input,
        out_path,
        input_names=["degraded_image"],
        output_names=["restored_image"],
        dynamic_axes={"degraded_image": {0: "batch", 2: "height", 3: "width"}, "restored_image": {0: "batch", 2: "height", 3: "width"}},
        opset_version=opset,
    )
    logger.info(f"Exported ONNX model to {out_path} (opset {opset}).")
    logger.info("Next steps for edge deployment: quantize with ONNX Runtime "
                "(dynamic or static INT8 quantization) or convert to TensorRT "
                "engine for NVIDIA edge hardware (e.g. Jetson-class controllers).")


if __name__ == "__main__":
    main()
