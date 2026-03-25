"""
Convert LabelMe JSON annotations to YOLO OBB format.

LabelMe JSON (per image):
    shapes: [{label, points: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]], shape_type: "polygon"}]
    imageWidth, imageHeight

YOLO OBB format (per line):
    class_id  x1 y1  x2 y2  x3 y3  x4 y4
    All coordinates normalized to [0, 1] by image dimensions.

Usage:
    python3 labelme_to_yolo_obb.py

Reads from RIVeR-Perception-Pipeline/labels/, writes to RIVeR-Perception-Pipeline/dataset/labels/.
"""

import json
import sys
from pathlib import Path

# Class mapping -- extend this when adding new objects
CLASS_MAP = {"banana": 0}

# Directories
SCRIPT_DIR = Path(__file__).parent
LABELS_INPUT_DIR = SCRIPT_DIR.parent / "labels"
LABELS_OUTPUT_DIR = SCRIPT_DIR.parent / "dataset" / "labels"


def normalize_points(
    points: list[list[float]], img_width: int, img_height: int
) -> list[list[float]]:
    """Normalize pixel coordinates to [0, 1] by image dimensions.

    Args:
        points: List of [x, y] coordinate pairs in pixel space.
        img_width: Image width in pixels.
        img_height: Image height in pixels.

    Returns:
        List of [x, y] pairs normalized to [0, 1].
    """
    return [[x / img_width, y / img_height] for x, y in points]


def labelme_shape_to_yolo_obb_line(
    shape: dict, img_width: int, img_height: int
) -> str:
    """Convert a single LabelMe shape to a YOLO OBB annotation line.

    Args:
        shape: LabelMe shape dict with 'label', 'points', 'shape_type'.
        img_width: Image width in pixels.
        img_height: Image height in pixels.

    Returns:
        String in format: "class_id x1 y1 x2 y2 x3 y3 x4 y4"

    Raises:
        ValueError: If label not in CLASS_MAP or points count != 4.
    """
    label = shape["label"]
    points = shape["points"]

    if label not in CLASS_MAP:
        raise ValueError(
            f"Unknown class label '{label}'. Known classes: {list(CLASS_MAP.keys())}"
        )

    if len(points) != 4:
        raise ValueError(
            f"OBB requires exactly 4 points, got {len(points)}. "
            f"Draw a 4-point polygon in LabelMe."
        )

    class_id = CLASS_MAP[label]
    normalized = normalize_points(points, img_width, img_height)

    # Format: class_id x1 y1 x2 y2 x3 y3 x4 y4
    coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in normalized)
    return f"{class_id} {coords}"


def convert_single_file(json_path: Path) -> list[str]:
    """Convert a single LabelMe JSON file to YOLO OBB format lines.

    Args:
        json_path: Path to LabelMe JSON file.

    Returns:
        List of YOLO OBB annotation lines (one per shape).
    """
    with open(json_path) as f:
        data = json.load(f)

    img_width = data["imageWidth"]
    img_height = data["imageHeight"]
    shapes = data.get("shapes", [])

    lines = []
    for shape in shapes:
        line = labelme_shape_to_yolo_obb_line(shape, img_width, img_height)
        lines.append(line)

    return lines


def main():
    """Convert all LabelMe JSON files in labels/ to YOLO OBB format."""
    LABELS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    json_files = sorted(LABELS_INPUT_DIR.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {LABELS_INPUT_DIR}")
        sys.exit(1)

    print(f"Converting {len(json_files)} files...")
    converted = 0
    errors = 0

    for json_path in json_files:
        try:
            lines = convert_single_file(json_path)
            # Write .txt file with same stem as the JSON
            txt_path = LABELS_OUTPUT_DIR / f"{json_path.stem}.txt"
            txt_path.write_text("\n".join(lines) + "\n" if lines else "")
            converted += 1
        except (ValueError, KeyError, json.JSONDecodeError) as e:
            print(f"  ERROR: {json_path.name}: {e}")
            errors += 1

    print(f"\nDone. {converted} converted, {errors} errors.")
    print(f"Output: {LABELS_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
