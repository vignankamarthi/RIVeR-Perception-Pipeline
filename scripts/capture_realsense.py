"""
RealSense D455 Capture Script
Runs natively on Mac. Saves RGB frames as PNGs.

Usage:
    python3 capture_realsense.py

Controls:
    SPACE or 's' -- save current frame
    'q'          -- quit
"""

import sys
from pathlib import Path

from capture_utils import get_next_index, make_filename

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "images" / "realsense"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PREFIX = "rs"


def main():
    # Hardware imports deferred to runtime -- keeps module importable for tests
    import cv2
    import numpy as np
    import pyrealsense2 as rs

    # Configure RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()

    # Enable RGB stream (1280x720 @ 30fps)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    # Start streaming
    try:
        pipeline.start(config)
    except RuntimeError as e:
        print(f"Failed to start RealSense: {e}")
        print("Is the camera plugged in?")
        sys.exit(1)

    print("RealSense D455 streaming.")
    print(f"Saving to: {OUTPUT_DIR}")
    print("Controls: SPACE/s = save frame, q = quit")
    print("-" * 40)

    frame_index = get_next_index(OUTPUT_DIR, PREFIX)
    saved_count = 0

    try:
        while True:
            # Wait for frames
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()

            if not color_frame:
                continue

            # Convert to numpy array
            color_image = np.asanyarray(color_frame.get_data())

            # Display with saved count overlay
            display = color_image.copy()
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
                cv2.imwrite(str(filename), color_image)
                print(f"  Saved: {filename.name}")
                frame_index += 1
                saved_count += 1

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print(f"\nDone. {saved_count} images saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
