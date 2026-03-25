"""
Azure Kinect DK Capture Script
Runs on Ubuntu lab machine (via SSH). Saves RGB frames as PNGs.

Usage:
    python3 capture_kinect.py

Controls:
    SPACE or 's' -- save current frame
    'q'          -- quit

If no display available (SSH without X forwarding), use --timed mode:
    python3 capture_kinect.py --timed 3
    Captures a frame every 3 seconds. Arrange banana between captures.
"""

import argparse
import sys
import time
from pathlib import Path

from capture_utils import get_next_index, make_filename

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "images" / "kinect"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PREFIX = "kt"


def parse_args():
    parser = argparse.ArgumentParser(description="Azure Kinect DK Capture")
    parser.add_argument(
        "--timed",
        type=int,
        default=0,
        help="Timed capture mode: capture a frame every N seconds (0 = interactive)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=50,
        help="Number of frames to capture in timed mode (default: 50)",
    )
    return parser.parse_args()


def main():
    # Hardware imports deferred to runtime
    import cv2
    import numpy as np
    from pyk4a import PyK4A, Config, ColorResolution

    args = parse_args()

    # Configure Kinect
    k4a = PyK4A(
        Config(
            color_resolution=ColorResolution.RES_1080P,
        )
    )

    try:
        k4a.start()
    except Exception as e:
        print(f"Failed to start Kinect: {e}")
        print("Is the camera plugged in?")
        sys.exit(1)

    print("Azure Kinect DK streaming.")
    print(f"Saving to: {OUTPUT_DIR}")
    print("-" * 40)

    frame_index = get_next_index(OUTPUT_DIR, PREFIX)
    saved_count = 0

    try:
        if args.timed > 0:
            # Timed capture mode (no display needed)
            print(f"Timed mode: capturing every {args.timed}s, target {args.count} frames")
            print("Arrange banana, then wait for capture. Ctrl+C to stop early.")
            for i in range(args.count):
                capture = k4a.get_capture()
                if capture.color is not None:
                    img = capture.color[:, :, :3]  # BGRA -> BGR
                    filename = OUTPUT_DIR / make_filename(PREFIX, frame_index)
                    cv2.imwrite(str(filename), img)
                    print(f"  [{saved_count + 1}/{args.count}] Saved: {filename.name}")
                    frame_index += 1
                    saved_count += 1
                time.sleep(args.timed)
        else:
            # Interactive mode (needs display)
            print("Controls: SPACE/s = save frame, q = quit")
            while True:
                capture = k4a.get_capture()
                if capture.color is None:
                    continue

                img = capture.color[:, :, :3]  # BGRA -> BGR

                display = img.copy()
                cv2.putText(
                    display,
                    f"Saved: {saved_count} | Next: {make_filename(PREFIX, frame_index)}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("Kinect DK - Press SPACE to capture, Q to quit", display)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord(" ") or key == ord("s"):
                    filename = OUTPUT_DIR / make_filename(PREFIX, frame_index)
                    cv2.imwrite(str(filename), img)
                    print(f"  Saved: {filename.name}")
                    frame_index += 1
                    saved_count += 1

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        k4a.stop()
        cv2.destroyAllWindows()
        print(f"\nDone. {saved_count} images saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
