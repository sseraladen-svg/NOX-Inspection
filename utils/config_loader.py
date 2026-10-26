from __future__ import annotations

from pathlib import Path
from typing import Any


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load a YAML configuration, or return the built-in defaults if omitted."""
    if path is None:
        return {
            "preprocessing": {"image_size": 640, "blur_kernel": 3},
            "detection": {
                "model_path": "models/yolo11n-seg.pt",
                "confidence": 0.5,
            },
            "classification": {
                "model_path": "models/classifier.onnx",
                "class_names": ["scratch", "dent", "discoloration"],
                "input_size": 224,
            },
            "anomaly_detection": {
                "memory_bank_path": "models/patchcore_model",
                "threshold": 0.5,
            },
        }

    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to load YAML configuration files") from exc

    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must contain a YAML mapping: {config_path}")
    return config
