from __future__ import annotations

import argparse
import math
import re
import sys
import threading
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from record.terminal_input import read_key
else:
    from .terminal_input import read_key

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

import numpy as np

try:
    from wakeforge.audio import (
        TARGET_SR,
        audio_stats,
        resample,
        save_wav,
        to_target_format,
    )
except ImportError as exc:
    raise SystemExit(
        f"Could not import 'wakeforge.audio' ({exc}).\n"
        'Install the project from its root with:  pip install -e ".[record]"'
    ) from exc

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = Path(__file__).resolve().parent / "config.toml"

CATEGORIES = ("wake_word", "negatives", "noise")

DEFAULT_CONFIG: dict[str, dict[str, Any]] = {
    "paths": {"output_dir": "my_recordings"},
    "device": {"input": "default", "output": "default"},
    "wake_word": {"min_seconds": 0.3, "max_seconds": 3.0, "default_takes": 50},
    "negatives": {"min_seconds": 0.3, "max_seconds": 5.0, "default_takes": 50},
    "noise": {"min_seconds": 5.0, "max_seconds": 600.0, "default_takes": 1},
}

SILENCE_PEAK_DBFS = -60.0
CLIPPING_FRACTION = 0.001


def load_config(path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    cfg = {section: dict(values) for section, values in DEFAULT_CONFIG.items()}
    cfg_path = Path(path) if path else CONFIG_PATH
    if cfg_path.is_file():
        with open(cfg_path, "rb") as fh:
            user = tomllib.load(fh)
        for section, values in user.items():
            if isinstance(values, dict):
                cfg.setdefault(section, {}).update(values)
    return cfg


def resolve_output_dir(cfg: dict, override: str | None = None) -> Path:
    if override:
        return Path(override).expanduser().resolve()
    p = Path(cfg["paths"]["output_dir"]).expanduser()
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def parse_device(value: Any) -> int | str | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if text == "" or text.lower() == "default":
        return None
    return int(text) if text.isdigit() else text


def _filename_regex(category: str) -> re.Pattern[str]:
    return re.compile(rf"^{re.escape(category)}_(\d+)\.wav$", re.IGNORECASE)


def count_existing(category_dir: Path, category: str) -> int:
    if not category_dir.is_dir():
        return 0
    rx = _filename_regex(category)
    return sum(1 for p in category_dir.iterdir() if rx.match(p.name))


def next_index(category_dir: Path, category: str) -> int:
    if not category_dir.is_dir():
        return 1
    rx = _filename_regex(category)
    indices = [
        int(m.group(1))
        for p in category_dir.iterdir()
        if (m := rx.match(p.name))
    ]
    return max(indices, default=0) + 1


def save_take(out_dir: Path, category: str, audio: np.ndarray) -> Path:
    category_dir = out_dir / category
    category_dir.mkdir(parents=True, exist_ok=True)
    idx = next_index(category_dir, category)
    for _ in range(1000):
        path = category_dir / f"{category}_{idx:04d}.wav"
        try:
            return save_wav(path, audio, TARGET_SR)
        except FileExistsError:
            idx += 1
    raise RuntimeError(f"No free filename found in {category_dir}")


def check_take(stats, min_seconds: float) -> list[str]:
    warnings: list[str] = []
    if stats.duration_s < min_seconds:
        warnings.append(
            f"too short ({stats.duration_s:.2f} s; recommended minimum {min_seconds:g} s)"
        )
    if stats.peak_dbfs < SILENCE_PEAK_DBFS:
        warnings.append(
            f"almost silent (peak {stats.peak_dbfs:.0f} dBFS): "
            "is the microphone muted or incorrectly selected?"
        )
    if stats.clipped_fraction > CLIPPING_FRACTION:
        warnings.append(
            "clipping: lower the microphone gain or move farther away"
        )
    return warnings


def import_sounddevice():
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise SystemExit(
            f"Could not import 'sounddevice' ({exc}).\n"
            'Install it with:  pip install -e ".[record]"\n'
            "On Linux, PortAudio is also required:  sudo apt install libportaudio2"
        ) from exc
    return sd


def resolve_input_rate(sd, device) -> int:
    try:
        sd.check_input_settings(
            device=device,
            channels=1,
            dtype="int16",
            samplerate=TARGET_SR,
        )
        return TARGET_SR
    except sd.PortAudioError:
        info = sd.query_devices(device, "input")
        return int(info["default_samplerate"])


class MicCapture:
    BAR_WIDTH = 16

    def __init__(self, sd, device, rate: int, max_seconds: float):
        self._sd = sd
        self.device = device
        self.rate = rate
        self.max_seconds = max_seconds
        self._max_frames = int(max_seconds * rate)
        self._chunks: list[np.ndarray] = []
        self._frames = 0
        self._peak = 0.0
        self._limit_hit = False
        self._stream = None
        self._thread: threading.Thread | None = None
        self._stop_evt = threading.Event()

    def _callback(self, indata, frames, time_info, status):
        remaining = self._max_frames - self._frames
        if remaining <= 0:
            self._limit_hit = True
            return
        block = indata[:remaining, 0].copy()
        self._chunks.append(block)
        self._frames += len(block)
        if len(block) < frames:
            self._limit_hit = True
        self._peak = max(
            self._peak,
            float(np.max(np.abs(block.astype(np.int32)))) / 32768.0,
        )

    def start(self) -> None:
        self._stream = self._sd.InputStream(
            samplerate=self.rate,
            channels=1,
            dtype="int16",
            device=self.device,
            callback=self._callback,
        )
        self._stream.start()
        self._stop_evt.clear()
        self._thread = threading.Thread(
            target=self._show_progress,
            daemon=True,
        )
        self._thread.start()

    def _show_progress(self) -> None:
        while not self._stop_evt.is_set():
            peak, self._peak = self._peak, 0.0
            db = 20.0 * math.log10(max(peak, 1e-6))
            filled = int(
                max(0.0, min(1.0, (db + 60.0) / 60.0)) * self.BAR_WIDTH
            )
            bar = "#" * filled + "-" * (self.BAR_WIDTH - filled)
            elapsed = self._frames / self.rate
            tail = (
                "LIMIT: press ENTER"
                if self._limit_hit
                else "ENTER = finish"
            )
            sys.stdout.write(
                f"\r   [REC] {elapsed:6.1f}s/{self.max_seconds:g}s "
                f"[{bar}] {db:6.1f} dBFS  {tail}   "
            )
            sys.stdout.flush()
            self._stop_evt.wait(0.1)

    def stop(self) -> np.ndarray:
        if self._stream is not None:
            try:
                self._stream.stop()
            finally:
                self._stream.close()
                self._stream = None
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        sys.stdout.write("\r" + " " * 78 + "\r")
        sys.stdout.flush()
        if not self._chunks:
            return np.zeros(0, dtype=np.int16)
        return np.concatenate(self._chunks)


def play_audio(sd, audio: np.ndarray, sr: int, device) -> None:
    try:
        try:
            sd.play(audio, sr, device=device)
        except sd.PortAudioError:
            info = sd.query_devices(device, "output")
            native = int(info["default_samplerate"])
            sd.play(
                resample(audio, sr, native),
                native,
                device=device,
            )
        sd.wait()
    except KeyboardInterrupt:
        sd.stop()


def wait_key(valid: set[str]) -> str:
    while True:
        key = read_key()
        if key in valid:
            return key


def run_session(sd, cfg, category, takes, out_dir, in_dev, out_dev) -> int:
    limits = cfg[category]
    min_s = float(limits["min_seconds"])
    max_s = float(limits["max_seconds"])

    try:
        rate = resolve_input_rate(sd, in_dev)
        dev_name = sd.query_devices(in_dev, "input")["name"]
    except Exception as exc:
        print(f"Could not access the microphone: {exc}")
        print("Use --list-devices to see the available devices.")
        return 1

    category_dir = out_dir / category
    category_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nCategory : {category}")
    print(f"Folder   : {category_dir}")
    print(f"Microphone: {dev_name}")
    if rate != TARGET_SR:
        print(
            f"            (capturing at {rate} Hz; "
            f"converted to {TARGET_SR} Hz when saving)"
        )
    print(
        f"Format   : WAV PCM16 mono {TARGET_SR} Hz | "
        f"maximum duration per take: {max_s:g} s"
    )

    done = 0
    last_audio: np.ndarray | None = None

    while done < takes:
        total = count_existing(category_dir, category)
        print(
            f"\n[{done + 1}/{takes}] {category}   "
            f"(files in folder: {total})"
        )
        print("   ENTER = start   Q = quit")
        if wait_key({"enter", "q"}) == "q":
            break

        capture = MicCapture(sd, in_dev, rate, max_s)
        try:
            capture.start()
        except Exception as exc:
            print(f"Could not start recording: {exc}")
            return 1

        try:
            wait_key({"enter"})
        finally:
            raw = capture.stop()

        if raw.size == 0:
            print("   No audio was captured; try again.")
            continue

        audio = to_target_format(raw, rate)
        stats = audio_stats(audio, TARGET_SR)
        path = save_take(out_dir, category, audio)
        done += 1
        last_audio = audio

        print(
            f"   saved {path.name}  "
            f"({stats.duration_s:.1f} s, peak {stats.peak_dbfs:.1f} dBFS)"
        )

        for warning in check_take(stats, min_s):
            print(f"   ! {warning}  -> press R to retake")

        quit_now = False

        while True:
            next_label = "finish" if done >= takes else "next"
            print(
                f"   [ENTER] {next_label}  "
                f"[R] retake  [P] play  [Q] quit"
            )

            key = wait_key({"enter", "r", "p", "q"})

            if key == "enter":
                break

            if key == "p":
                play_audio(sd, last_audio, TARGET_SR, out_dev)

            elif key == "r":
                path.unlink(missing_ok=True)
                print(f"   deleted {path.name}; recording again")
                done -= 1
                last_audio = None
                break

            elif key == "q":
                quit_now = True
                break

        if quit_now:
            break

    print(
        f"\nSession finished: {done} take(s) saved in {category_dir}"
    )
    return 0


def ask_category() -> str:
    print("What do you want to record?")

    for i, name in enumerate(CATEGORIES, 1):
        print(f"  {i}) {name}")

    while True:
        raw = input("Choose 1-3 or enter the name: ").strip().lower()

        if raw.isdigit() and 1 <= int(raw) <= len(CATEGORIES):
            return CATEGORIES[int(raw) - 1]

        if raw in CATEGORIES:
            return raw

        print("Invalid option.")


def ask_takes(default: int) -> int:
    while True:
        raw = input(f"How many takes? [{default}]: ").strip()

        if not raw:
            return default

        if raw.isdigit() and int(raw) > 0:
            return int(raw)

        print("Enter an integer greater than 0.")


def positive_int(value: str) -> int:
    n = int(value)

    if n <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")

    return n


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m record.recorder",
        description="Record the WakeForge dataset from the terminal.",
    )

    p.add_argument(
        "-c",
        "--category",
        choices=CATEGORIES,
        help="type of audio to record",
    )

    p.add_argument(
        "-n",
        "--takes",
        type=positive_int,
        help="number of takes in the session",
    )

    p.add_argument(
        "--config",
        help=f"path to a config.toml file (default: {CONFIG_PATH})",
    )

    p.add_argument(
        "--output-dir",
        help="output folder (default: the one from config.toml)",
    )

    p.add_argument(
        "--input-device",
        help="microphone index or name",
    )

    p.add_argument(
        "--output-device",
        help="speaker index or name",
    )

    p.add_argument(
        "--list-devices",
        action="store_true",
        help="list audio devices and exit",
    )

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    sd = import_sounddevice()

    if args.list_devices:
        print(sd.query_devices())
        return 0

    in_dev = parse_device(
        args.input_device
        if args.input_device is not None
        else cfg["device"]["input"]
    )

    out_dev = parse_device(
        args.output_device
        if args.output_device is not None
        else cfg["device"]["output"]
    )

    out_dir = resolve_output_dir(cfg, args.output_dir)

    try:
        category = args.category or ask_category()
        takes = args.takes or ask_takes(
            int(cfg[category].get("default_takes", 50))
        )
        return run_session(
            sd,
            cfg,
            category,
            takes,
            out_dir,
            in_dev,
            out_dev,
        )

    except (KeyboardInterrupt, EOFError):
        print("\nInterrupted. Already saved takes are preserved.")
        return 130

    finally:
        sd.stop()


if __name__ == "__main__":
    sys.exit(main())