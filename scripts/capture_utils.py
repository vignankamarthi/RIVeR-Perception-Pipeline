"""
Shared utilities for capture scripts.
Pure logic -- no hardware dependencies (no cv2, no pyrealsense2, no pyk4a).
"""

from pathlib import Path


def get_next_index(output_dir: Path, prefix: str) -> int:
    """Find the next available index based on existing files.

    Looks for files matching {prefix}_NNN.png and returns max(NNN) + 1.
    Returns 1 if no matching files exist.

    Args:
        output_dir: Directory to scan for existing files.
        prefix: Filename prefix (e.g., "rs" for rs_001.png, "kt" for kt_001.png).

    Returns:
        Next available integer index.
    """
    existing = list(output_dir.glob(f"{prefix}_*.png"))
    if not existing:
        return 1
    indices = []
    for f in existing:
        try:
            idx = int(f.stem.split("_")[1])
            indices.append(idx)
        except (ValueError, IndexError):
            continue
    return max(indices) + 1 if indices else 1


def make_filename(prefix: str, index: int) -> str:
    """Generate a zero-padded filename.

    Args:
        prefix: Filename prefix (e.g., "rs", "kt").
        index: Integer index.

    Returns:
        Filename string like "rs_001.png".
    """
    return f"{prefix}_{index:03d}.png"
