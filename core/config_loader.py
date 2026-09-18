"""
core/config_loader.py

Loads a product config (config/<product_id>.yaml) and checks required keys
are present per config/schema.yaml. Raises on missing keys instead of
failing silently deep inside a pipeline stage.
"""

import os
import yaml

REQUIRED_TOP_KEYS = [
    "product_id",
    "stations",
    "cameras_per_station",
    "reorientation_steps",
    "roi",
    "preprocessing",
    "models",
    "defect_classes",
    "decision_rules",
]

REQUIRED_PREPROCESSING_KEYS = ["resize", "blur_kernel", "clahe"]
REQUIRED_MODEL_KEYS = ["yolo_seg", "cnn_classifier", "anomaly"]
REQUIRED_DECISION_KEYS = ["fail_confidence_threshold", "severity_map"]


def load_config(product_id: str, config_dir: str = "config") -> dict:
    path = os.path.join(config_dir, f"{product_id}.yaml")

    if not os.path.exists(path):
        raise FileNotFoundError(f"No config found for product '{product_id}' at {path}")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    _validate(config, path)
    return config


def _validate(config: dict, path: str) -> None:
    missing = [k for k in REQUIRED_TOP_KEYS if k not in config]
    if missing:
        raise ValueError(f"{path}: missing required top-level key(s): {missing}")

    missing_pre = [k for k in REQUIRED_PREPROCESSING_KEYS if k not in config["preprocessing"]]
    if missing_pre:
        raise ValueError(f"{path}: missing preprocessing key(s): {missing_pre}")

    missing_models = [k for k in REQUIRED_MODEL_KEYS if k not in config["models"]]
    if missing_models:
        raise ValueError(f"{path}: missing model key(s): {missing_models}")

    missing_decision = [k for k in REQUIRED_DECISION_KEYS if k not in config["decision_rules"]]
    if missing_decision:
        raise ValueError(f"{path}: missing decision_rules key(s): {missing_decision}")
