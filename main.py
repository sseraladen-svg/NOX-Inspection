import argparse
from pathlib import Path

import cv2

from pipeline.router import PipelineRouter
from utils.config_loader import load_config
from utils.overlay import draw_aoi_overlay


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local AOI inspection")
    parser.add_argument("--image", type=Path, default=Path("test_battery.jpg"))
    parser.add_argument("--product", default="prismatic_cell")
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()

    if not args.image.is_file():
        raise FileNotFoundError(f"Inspection image not found: {args.image}")
    config_path = args.config or Path("config") / f"{args.product}.yaml"
    if config_path.is_file():
        config = load_config(config_path)
    else:
        print(f"WARNING: {config_path} not found; using built-in pipeline defaults.")
        config = load_config()

    frame = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f"Could not read inspection image: {args.image}")

    router = PipelineRouter(config)
    decision, timings = router.run(frame, args.product, "local")
    processed = decision.pop("_processed_frame")
    annotated = draw_aoi_overlay(processed, decision["detections"])

    print(
        f"{decision['result']} | Defect: {decision['defect_type']} | "
        f"Confidence: {decision['confidence']:.2f}"
    )
    print("Stage timings (ms): " + ", ".join(
        f"{stage}={duration:.2f}" for stage, duration in timings.items()
    ))
    cv2.imshow("BatteryVisionAI Inspection", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
