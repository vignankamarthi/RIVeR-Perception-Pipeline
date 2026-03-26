"""
Auto-label images using YOLO-World + SAM2.

Offline batch labeling: processes saved images on disk, generates annotation files.
NOT live labeling -- no camera interaction.

Pipeline:
    1. YOLO-World (yolov8x-worldv2.pt) detects objects via text prompt
    2. SAM2 (sam2_b.pt) segments detected regions into binary masks
    3. mask_to_obb converts masks to oriented bounding boxes
    4. Outputs both LabelMe JSON (for review) and YOLO OBB txt (for training)

Usage:
    python3 auto_label.py
    python3 auto_label.py --classes banana pear
    python3 auto_label.py --conf 0.3

Zero additional pip installs -- YOLO-World and SAM2 are both in ultralytics.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Pure logic functions (tested by test_auto_label.py)
# ---------------------------------------------------------------------------

def mask_to_obb(mask: np.ndarray, min_area: int = 100) -> np.ndarray | None:
    """Convert a binary mask to an oriented bounding box (4 corner points).

    Args:
        mask: Binary mask (H, W), uint8, values 0 or 1.
        min_area: Minimum contour area in pixels. Below this, return None.

    Returns:
        (4, 2) float32 array of corner points, or None if mask is empty/too small.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Use the largest contour
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < min_area:
        return None

    rect = cv2.minAreaRect(largest)
    box = cv2.boxPoints(rect)
    return box.astype(np.float32)


def obb_to_yolo_line(box: np.ndarray, class_id: int, img_w: int, img_h: int) -> str:
    """Convert OBB corner points to a YOLO OBB format line.

    Args:
        box: (4, 2) array of corner points in pixel coordinates.
        class_id: Integer class ID.
        img_w: Image width for normalization.
        img_h: Image height for normalization.

    Returns:
        String: "class_id x1 y1 x2 y2 x3 y3 x4 y4" with normalized coords.
    """
    coords = []
    for x, y in box:
        coords.append(f"{x / img_w:.6f}")
        coords.append(f"{y / img_h:.6f}")
    return f"{class_id} " + " ".join(coords)


def obb_to_labelme_shape(box: np.ndarray, label: str) -> dict:
    """Convert OBB corner points to a LabelMe shape dict.

    Args:
        box: (4, 2) array of corner points in pixel coordinates.
        label: Class label string.

    Returns:
        LabelMe shape dict with all required fields.
    """
    return {
        "label": label,
        "points": [[float(x), float(y)] for x, y in box],
        "shape_type": "polygon",
        "group_id": None,
        "description": "",
        "flags": {},
    }


def make_labelme_json(
    img_path: Path, shapes: list[dict], img_h: int, img_w: int
) -> dict:
    """Build a complete LabelMe JSON structure.

    Args:
        img_path: Path to the image file (used for imagePath field).
        shapes: List of LabelMe shape dicts.
        img_h: Image height.
        img_w: Image width.

    Returns:
        Dict ready to be written as JSON.
    """
    return {
        "version": "5.4.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": str(img_path),
        "imageData": None,
        "imageHeight": img_h,
        "imageWidth": img_w,
    }


# ---------------------------------------------------------------------------
# Orchestration functions (tested without models)
# ---------------------------------------------------------------------------

def process_detections(
    masks: list[np.ndarray],
    class_names: list[str],
    class_map: dict[str, int],
    img_w: int,
    img_h: int,
) -> tuple[list[dict], list[str]]:
    """Process detection masks into both LabelMe shapes and YOLO OBB lines.

    Takes pre-computed masks (no model inference) and converts them.

    Args:
        masks: List of binary masks from SAM2.
        class_names: Class name for each mask (parallel list).
        class_map: Mapping from class name to integer ID.
        img_w: Image width.
        img_h: Image height.

    Returns:
        Tuple of (labelme_shapes, yolo_lines).
    """
    shapes = []
    yolo_lines = []

    for mask, name in zip(masks, class_names):
        box = mask_to_obb(mask)
        if box is None:
            continue

        shapes.append(obb_to_labelme_shape(box, name))
        yolo_lines.append(obb_to_yolo_line(box, class_map[name], img_w, img_h))

    return shapes, yolo_lines


def save_labels(
    img_path: Path,
    shapes: list[dict],
    yolo_lines: list[str],
    labels_dir: Path,
    yolo_dir: Path,
    img_h: int,
    img_w: int,
) -> None:
    """Write LabelMe JSON and YOLO OBB txt files to disk.

    Args:
        img_path: Image filename (stem used for output filenames).
        shapes: LabelMe shape dicts.
        yolo_lines: YOLO OBB format lines.
        labels_dir: Directory for LabelMe JSON output.
        yolo_dir: Directory for YOLO OBB txt output.
        img_h: Image height.
        img_w: Image width.
    """
    labels_dir.mkdir(parents=True, exist_ok=True)
    yolo_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(img_path).stem

    # Write LabelMe JSON
    labelme_data = make_labelme_json(img_path, shapes, img_h, img_w)
    json_path = labels_dir / f"{stem}.json"
    json_path.write_text(json.dumps(labelme_data, indent=2) + "\n")

    # Write YOLO OBB txt
    txt_path = yolo_dir / f"{stem}.txt"
    txt_path.write_text("\n".join(yolo_lines) + "\n" if yolo_lines else "")


# ---------------------------------------------------------------------------
# Inference functions (hardware-dependent, tested manually)
# ---------------------------------------------------------------------------

def load_models(
    det_weights: str, sam_weights: str, classes: list[str]
) -> tuple:
    """Load YOLO-World detection model and SAM2 segmentation model.

    Args:
        det_weights: Path or name of YOLO-World weights.
        sam_weights: Path or name of SAM2 weights.
        classes: List of class names for open-vocabulary detection.

    Returns:
        Tuple of (det_model, sam_model).
    """
    from ultralytics import YOLO, SAM

    det_model = YOLO(det_weights)
    det_model.set_classes(classes)

    sam_model = SAM(sam_weights)

    return det_model, sam_model


def auto_label_image(
    img_path: Path,
    det_model,
    sam_model,
    classes: list[str],
    class_map: dict[str, int],
    conf: float = 0.25,
) -> tuple[list[dict], list[str], list[float]]:
    """Auto-label a single image: detect, segment, convert to OBB.

    Args:
        img_path: Path to the image.
        det_model: Loaded YOLO-World model.
        sam_model: Loaded SAM2 model.
        classes: List of class names.
        class_map: Class name to integer ID mapping.
        conf: Detection confidence threshold.

    Returns:
        Tuple of (labelme_shapes, yolo_lines, confidences).
    """
    # Step 1: Detect with YOLO-World
    det_results = det_model(str(img_path), conf=conf, verbose=False)
    det = det_results[0]

    if len(det.boxes) == 0:
        return [], [], []

    # Extract bounding boxes and class info
    bboxes = det.boxes.xyxy.cpu().numpy()
    cls_ids = det.boxes.cls.cpu().numpy().astype(int)
    confs = det.boxes.conf.cpu().numpy().tolist()
    class_names = [classes[cid] for cid in cls_ids]

    # Step 2: Segment with SAM2 using detection bounding boxes
    sam_results = sam_model(str(img_path), bboxes=bboxes, verbose=False)

    # Extract masks
    masks = []
    if sam_results[0].masks is not None:
        for mask_tensor in sam_results[0].masks.data:
            mask_np = mask_tensor.cpu().numpy().astype(np.uint8)
            masks.append(mask_np)

    # Match masks to class names (SAM returns one mask per bbox)
    # If fewer masks than detections, only process what we have
    n = min(len(masks), len(class_names))
    masks = masks[:n]
    class_names = class_names[:n]

    # Step 3: Convert masks to OBBs and format outputs
    img = cv2.imread(str(img_path))
    img_h, img_w = img.shape[:2]
    shapes, yolo_lines = process_detections(masks, class_names, class_map, img_w, img_h)

    return shapes, yolo_lines, confs[:n]


def main():
    """CLI entry point: auto-label all images in a directory."""
    parser = argparse.ArgumentParser(
        description="Auto-label images with YOLO-World + SAM2 (offline batch)."
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path(__file__).parent.parent / "images",
        help="Directory containing image subdirectories (default: ../images)",
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=Path(__file__).parent.parent / "labels",
        help="Output directory for LabelMe JSON (default: ../labels)",
    )
    parser.add_argument(
        "--yolo-dir",
        type=Path,
        default=Path(__file__).parent.parent / "dataset" / "labels",
        help="Output directory for YOLO OBB txt (default: ../dataset/labels)",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=["banana"],
        help="Object classes to detect (default: banana)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Detection confidence threshold (default: 0.25)",
    )
    parser.add_argument(
        "--det-weights",
        default="yolov8x-worldv2.pt",
        help="YOLO-World model weights (default: yolov8x-worldv2.pt)",
    )
    parser.add_argument(
        "--sam-weights",
        default="sam2_b.pt",
        help="SAM2 model weights (default: sam2_b.pt)",
    )
    args = parser.parse_args()

    # Build class map
    class_map = {name: i for i, name in enumerate(args.classes)}
    print(f"Classes: {class_map}")
    print(f"Confidence threshold: {args.conf}")
    print(f"Images dir: {args.images_dir}")

    # Collect all images
    image_paths = []
    for subdir in sorted(args.images_dir.iterdir()):
        if subdir.is_dir():
            image_paths.extend(sorted(subdir.glob("*.png")))
    # Also check for images directly in the root
    image_paths.extend(sorted(args.images_dir.glob("*.png")))

    if not image_paths:
        print(f"No PNG images found in {args.images_dir}")
        sys.exit(1)

    print(f"Found {len(image_paths)} images.")
    print()

    # Load models
    print("Loading YOLO-World model...")
    det_model, sam_model = load_models(args.det_weights, args.sam_weights, args.classes)
    print("Models loaded.\n")

    # Process each image
    total_detections = 0
    zero_detection_images = []
    all_confidences = []
    per_image_stats = []

    for i, img_path in enumerate(image_paths, 1):
        shapes, yolo_lines, confs = auto_label_image(
            img_path, det_model, sam_model, args.classes, class_map, args.conf
        )

        # Read image dimensions for save
        img = cv2.imread(str(img_path))
        img_h, img_w = img.shape[:2]

        # Save outputs
        save_labels(img_path, shapes, yolo_lines, args.labels_dir, args.yolo_dir, img_h, img_w)

        n_det = len(shapes)
        total_detections += n_det
        all_confidences.extend(confs)

        if n_det == 0:
            zero_detection_images.append(img_path.name)

        per_image_stats.append((img_path.name, n_det, confs))
        status = f"[{i}/{len(image_paths)}] {img_path.name}: {n_det} detection(s)"
        if confs:
            status += f"  conf: {min(confs):.2f}-{max(confs):.2f}"
        print(status)

    # Summary
    print("\n" + "=" * 60)
    print("AUTO-LABELING COMPLETE")
    print("=" * 60)
    print(f"Images processed: {len(image_paths)}")
    print(f"Total detections: {total_detections}")
    print(f"Images with 0 detections: {len(zero_detection_images)}")
    if zero_detection_images:
        print(f"  Missing: {', '.join(zero_detection_images)}")
    if all_confidences:
        print(f"Confidence range: {min(all_confidences):.3f} - {max(all_confidences):.3f}")
        print(f"Confidence mean: {sum(all_confidences) / len(all_confidences):.3f}")
    print(f"\nLabelMe JSON: {args.labels_dir}")
    print(f"YOLO OBB txt: {args.yolo_dir}")


if __name__ == "__main__":
    main()
