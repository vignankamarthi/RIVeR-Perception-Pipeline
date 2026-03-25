"""
Visualize YOLO OBB labels overlaid on images for spot-checking.

Usage:
    python3 visualize_labels.py
    python3 visualize_labels.py --count 5
    python3 visualize_labels.py --image-dir path/to/images --label-dir path/to/labels
"""

import argparse
import random
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
DEFAULT_IMAGE_DIR = SCRIPT_DIR.parent / "dataset" / "images" / "train"
DEFAULT_LABEL_DIR = SCRIPT_DIR.parent / "dataset" / "labels" / "train"

# Reverse class map for display
CLASS_NAMES = {0: "banana"}


def read_yolo_obb_label(label_path: Path) -> list[dict]:
    """Parse a YOLO OBB label file into structured data.

    Args:
        label_path: Path to .txt label file.

    Returns:
        List of dicts with 'class_id', 'class_name', 'points' keys.
        Points are normalized [0,1] as [[x1,y1], [x2,y2], [x3,y3], [x4,y4]].
    """
    labels = []
    text = label_path.read_text().strip()
    if not text:
        return labels

    for line in text.split("\n"):
        parts = line.strip().split()
        if len(parts) != 9:
            continue
        class_id = int(parts[0])
        coords = [float(x) for x in parts[1:]]
        points = [[coords[i], coords[i + 1]] for i in range(0, 8, 2)]
        labels.append({
            "class_id": class_id,
            "class_name": CLASS_NAMES.get(class_id, f"class_{class_id}"),
            "points": points,
        })
    return labels


def denormalize_points(
    points: list[list[float]], img_width: int, img_height: int
) -> list[list[int]]:
    """Convert normalized [0,1] points back to pixel coordinates.

    Args:
        points: Normalized coordinate pairs.
        img_width: Image width in pixels.
        img_height: Image height in pixels.

    Returns:
        Pixel coordinate pairs as integers.
    """
    return [[int(x * img_width), int(y * img_height)] for x, y in points]


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize YOLO OBB labels")
    parser.add_argument("--image-dir", type=str, default=str(DEFAULT_IMAGE_DIR))
    parser.add_argument("--label-dir", type=str, default=str(DEFAULT_LABEL_DIR))
    parser.add_argument("--count", type=int, default=3, help="Number of images to show")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for selection")
    return parser.parse_args()


def main():
    # Deferred import -- cv2 + numpy only needed at runtime
    import cv2
    import numpy as np

    args = parse_args()
    image_dir = Path(args.image_dir)
    label_dir = Path(args.label_dir)

    images = sorted(image_dir.glob("*.png"))
    if not images:
        print(f"No images found in {image_dir}")
        sys.exit(1)

    rng = random.Random(args.seed)
    selected = rng.sample(images, min(args.count, len(images)))

    for img_path in selected:
        label_path = label_dir / f"{img_path.stem}.txt"
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  Could not read: {img_path.name}")
            continue

        h, w = img.shape[:2]
        labels = read_yolo_obb_label(label_path) if label_path.exists() else []

        for lbl in labels:
            pts = denormalize_points(lbl["points"], w, h)
            pts_np = np.array(pts, dtype=np.int32)
            cv2.polylines(img, [pts_np], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.putText(
                img, lbl["class_name"], (pts[0][0], pts[0][1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
            )

        label_count = len(labels)
        title = f"{img_path.name} ({label_count} labels)"
        cv2.imshow(title, img)
        print(f"  Showing: {title} -- press any key for next")
        cv2.waitKey(0)

    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
