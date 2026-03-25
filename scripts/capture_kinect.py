"""
Azure Kinect DK Capture Script
Runs on Ubuntu lab machine (via SSH). Saves RGB frames as PNGs.

Usage:
    python3 capture_kinect.py                  # ENTER to capture (SSH-friendly, no GUI)
    python3 capture_kinect.py --timed 3        # auto-capture every 3 seconds
    python3 capture_kinect.py --gui            # interactive with preview window (needs display)

Controls (default mode):
    ENTER  -- save current frame
    'q' + ENTER -- quit
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
        help="Auto-capture every N seconds (0 = manual ENTER mode)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=50,
        help="Max frames to capture (default: 50)",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Interactive mode with preview window (needs display, not for SSH)",
    )
    return parser.parse_args()


def main():
    # Hardware imports deferred to runtime
    import cv2
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
            # Timed capture mode
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

        elif args.gui:
            # GUI interactive mode (needs display)
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

        else:
            # Default: ENTER-to-capture mode (SSH-friendly, no GUI)
            print("Press ENTER to capture, 'q' + ENTER to quit.")
            while saved_count < args.count:
                user_input = input(f"  [{saved_count}/{args.count}] ENTER to capture > ").strip()
                if user_input.lower() == "q":
                    break

                capture = k4a.get_capture()
                if capture.color is not None:
                    img = capture.color[:, :, :3]  # BGRA -> BGR
                    filename = OUTPUT_DIR / make_filename(PREFIX, frame_index)
                    cv2.imwrite(str(filename), img)
                    print(f"    Saved: {filename.name}")
                    frame_index += 1
                    saved_count += 1
                else:
                    print("    No frame received, try again.")

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        k4a.stop()
        print(f"\nDone. {saved_count} images saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
