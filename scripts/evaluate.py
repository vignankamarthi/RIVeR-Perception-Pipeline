"""
Run inference on validation images and save results for visual inspection.

Usage:
    python3 evaluate.py
    python3 evaluate.py --weights path/to/best.pt
    python3 evaluate.py --source path/to/images/
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

SCRIPT_DIR = Path(__file__).parent
DEFAULT_WEIGHTS = SCRIPT_DIR.parent / "models" / "banana_obb" / "weights" / "best.pt"
DEFAULT_SOURCE = SCRIPT_DIR.parent / "dataset" / "images" / "val"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate YOLO OBB model")
    parser.add_argument(
        "--weights",
        type=str,
        default=str(DEFAULT_WEIGHTS),
        help=f"Path to trained weights (default: {DEFAULT_WEIGHTS})",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=str(DEFAULT_SOURCE),
        help=f"Path to images to run inference on (default: {DEFAULT_SOURCE})",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Confidence threshold (default: 0.5)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    weights = Path(args.weights)
    source = Path(args.source)

    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}. Run train.py first.")
    if not source.exists():
        raise FileNotFoundError(f"Source not found: {source}. Run split_dataset.py first.")

    model = YOLO(str(weights))
    results = model.predict(str(source), save=True, conf=args.conf)

    print(f"\nInference complete. Check saved images for visual confirmation.")
    print(f"Look for: tight OBBs, correct rotation, no false positives/negatives.")
    return results


if __name__ == "__main__":
    main()
