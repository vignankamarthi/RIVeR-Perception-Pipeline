"""
Train YOLOv8-OBB on the banana dataset.

Usage:
    python3 train.py
"""

from pathlib import Path

from ultralytics import YOLO

SCRIPT_DIR = Path(__file__).parent
DATA_YAML = SCRIPT_DIR.parent / "dataset" / "data.yaml"
PROJECT_DIR = SCRIPT_DIR.parent / "models"


def main():
    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"data.yaml not found at {DATA_YAML}. Run split_dataset.py first."
        )

    model = YOLO("yolov8n-obb.pt")  # pretrained nano OBB model

    results = model.train(
        data=str(DATA_YAML),
        epochs=50,
        imgsz=640,
        batch=16,
        project=str(PROJECT_DIR),
        name="banana_obb",
    )

    print(f"\nTraining complete. Best weights: {PROJECT_DIR}/banana_obb/weights/best.pt")
    return results


if __name__ == "__main__":
    main()
