import argparse
import subprocess
import sys

from wakeforge import load_config, train


def main():
    ap = argparse.ArgumentParser(
        description="Train a wake word model and export it to ONNX."
    )

    ap.add_argument(
        "--config",
        default="config/config.toml",
        help="path to the TOML file",
    )

    ap.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="section.option=value",
        help="override an option; can be repeated",
    )

    args = ap.parse_args()

    config = load_config(args.config, args.set)
    train(config)

    print()
    print("=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)
    print("Opening microphone test...")
    print()

    subprocess.run(
        [sys.executable, "tests/test_onnx.py"],
        check=True,
    )


if __name__ == "__main__":
    main()