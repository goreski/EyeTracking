"""Driver Monitoring System (DMM Project) - Main Entry Point.

Usage:
    python main.py                     # Runs live camera stream on default camera (index 0)
    python main.py --camera 1          # Runs live camera stream on camera index 1
    python main.py --basler            # Runs with Basler USB3 Vision / GenICam camera
    python main.py --test              # Runs offline probabilistic pipeline test suite
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path
SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probabilistic Driver Attention and Drowsiness Monitoring System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera index for UVC webcam")
    parser.add_argument("--basler", action="store_true", help="Force Basler pypylon camera capture")
    parser.add_argument("--test", action="store_true", help="Run probabilistic ML test suite")
    return parser.parse_args()


def main():
    args = parse_arguments()

    if args.test:
        print("[INFO] Running probabilistic pipeline test suite...")
        import subprocess
        test_script = Path(__file__).resolve().parent / "tests" / "test_probabilistic_pipeline.py"
        result = subprocess.run([sys.executable, str(test_script)])
        sys.exit(result.returncode)

    if args.basler:
        os.environ["USE_BASLER_CAMERA"] = "true"

    from camera_stream import start_camera_stream
    start_camera_stream(camera_index=args.camera)


if __name__ == "__main__":
    main()
