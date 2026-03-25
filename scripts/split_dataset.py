"""
Split labeled dataset into train/val sets.

Reads images and YOLO OBB label files, splits 80/20, copies into
the dataset directory structure expected by Ultralytics.

Usage:
    python3 split_dataset.py
"""

import random
import shutil
import sys
from pathlib import Path

# Directories
SCRIPT_DIR = Path(__file__).parent
IMAGES_DIR = SCRIPT_DIR.parent / "images"
LABELS_DIR = SCRIPT_DIR.parent / "dataset" / "labels"  # output from conversion
DATASET_DIR = SCRIPT_DIR.parent / "dataset"


def compute_split(
    stems: list[str], val_ratio: float = 0.2, seed: int = 42
) -> tuple[list[str], list[str]]:
    """Compute train/val split on file stems.

    Args:
        stems: List of file stems (no extension).
        val_ratio: Fraction of data for validation (default 0.2).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_stems, val_stems).
    """
    if not stems:
        return [], []

    rng = random.Random(seed)
    shuffled = stems.copy()
    rng.shuffle(shuffled)

    val_count = max(0, int(len(shuffled) * val_ratio))
    val_stems = shuffled[:val_count]
    train_stems = shuffled[val_count:]

    return train_stems, val_stems


def execute_split(
    train_stems: list[str],
    val_stems: list[str],
    images_src: Path,
    labels_src: Path,
    output_base: Path,
    img_ext: str = ".png",
    label_ext: str = ".txt",
) -> None:
    """Copy files into train/val directory structure.

    Args:
        train_stems: File stems for training set.
        val_stems: File stems for validation set.
        images_src: Source directory for images.
        labels_src: Source directory for label files.
        output_base: Base output directory (will contain images/ and labels/).
        img_ext: Image file extension.
        label_ext: Label file extension.
    """
    for split_name, stems in [("train", train_stems), ("val", val_stems)]:
        img_dst = output_base / "images" / split_name
        lbl_dst = output_base / "labels" / split_name
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)

        for stem in stems:
            img_src_path = images_src / f"{stem}{img_ext}"
            lbl_src_path = labels_src / f"{stem}{label_ext}"

            if img_src_path.exists():
                shutil.copy2(img_src_path, img_dst / f"{stem}{img_ext}")
            if lbl_src_path.exists():
                shutil.copy2(lbl_src_path, lbl_dst / f"{stem}{label_ext}")


def main():
    """Run the split on the RIVeR-Perception-Pipeline dataset."""
    # Collect all image stems from both camera dirs
    all_images = []
    for subdir in ["realsense", "kinect"]:
        camera_dir = IMAGES_DIR / subdir
        if camera_dir.exists():
            all_images.extend(camera_dir.glob("*.png"))

    if not all_images:
        print(f"No images found in {IMAGES_DIR}/realsense/ or {IMAGES_DIR}/kinect/")
        sys.exit(1)

    stems = [f.stem for f in all_images]
    print(f"Found {len(stems)} images.")

    train_stems, val_stems = compute_split(stems, val_ratio=0.2, seed=42)
    print(f"Split: {len(train_stems)} train, {len(val_stems)} val")

    execute_split(
        train_stems=train_stems,
        val_stems=val_stems,
        images_src=IMAGES_DIR,
        labels_src=LABELS_DIR,
        output_base=DATASET_DIR,
    )

    print(f"Done. Output: {DATASET_DIR}")


if __name__ == "__main__":
    main()
