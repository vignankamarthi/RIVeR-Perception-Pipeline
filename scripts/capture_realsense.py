"""
RealSense D455 Capture Script
Runs natively on Mac via OpenCV (UVC mode). Saves RGB frames as PNGs.

Note: pyrealsense2-macosx segfaults on Apple Silicon (libusb issue).
The RealSense D455 exposes itself as a standard UVC camera, so OpenCV
can capture RGB frames directly. Depth is not available in this mode,
but RGB is sufficient for YOLO OBB training.

Usage:
    python3 capture_realsense.py
    python3 capture_realsense.py --camera 1    # specify camera index

Controls:
    SPACE or 's' -- save current frame
    'q'          -- quit
"""

import argparse
import sys
from pathlib import Path

from capture_utils import get_next_index, make_filename

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "images" / "realsense"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PREFIX = "rs"


def parse_args():
    parser = argparse.ArgumentParser(description="RealSense D455 Capture (OpenCV)")
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index (default: 1). Run with --list to see available cameras.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available cameras and exit.",
    )
    return parser.parse_args()


def list_cameras():
    """Detect and print available cameras."""
    import cv2

    print("Scanning cameras...")
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"  Camera {i}: {frame.shape[1]}x{frame.shape[0]}")
            cap.release()
    print("Done.")


def main():
    import cv2

    args = parse_args()

    if args.list:
        list_cameras()
        return

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Failed to open camera {args.camera}.")
        print("Run with --list to see available cameras.")
        sys.exit(1)

    # Get actual resolution
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"RealSense D455 streaming via OpenCV (camera {args.camera}).")
    print(f"Resolution: {width}x{height}")
    print(f"Saving to: {OUTPUT_DIR}")
    print("Controls: SPACE/s = save frame, q = quit")
    print("-" * 40)

    frame_index = get_next_index(OUTPUT_DIR, PREFIX)
    saved_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            # Display with saved count overlay
            display = frame.copy()
            cv2.putText(
                display,
                f"Saved: {saved_count} | Next: {make_filename(PREFIX, frame_index)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.imshow("RealSense D455 - Press SPACE to capture, Q to quit", display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord(" ") or key == ord("s"):
                filename = OUTPUT_DIR / make_filename(PREFIX, frame_index)
                cv2.imwrite(str(filename), frame)
                print(f"  Saved: {filename.name}")
                frame_index += 1
                saved_count += 1

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\nDone. {saved_count} images saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
