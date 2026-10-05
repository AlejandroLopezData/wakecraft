"""WakeForge · grabador de dataset desde terminal.

Uso (desde la raíz del proyecto):

    python -m record.recorder
    python -m record.recorder --category wake_word --takes 50
    python -m record.recorder --list-devices

Guarda WAV PCM16 mono 16 kHz en ``my_recordings/<categoria>/`` sin sobrescribir
nunca archivos existentes.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import threading
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # ejecutado como `python record/recorder.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from record.terminal_input import read_key
else:
    from .terminal_input import read_key

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]

import numpy as np

try:
    from wakeforge.audio import (
        TARGET_SR,
        audio_stats,
        resample,
        save_wav,
        to_target_format,
    )
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        f"No se pudo importar 'wakeforge.audio' ({exc}).\n"
        'Instala el proyecto desde su raíz con:  pip install -e ".[record]"'
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


# --------------------------------------------------------------------------
# Lógica pura (sin hardware): configuración, nombres, validación
# --------------------------------------------------------------------------
def load_config(path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    """Carga record/config.toml sobre los valores por defecto."""
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
    """--output-dir es relativo al directorio actual; el de config.toml, a la raíz del proyecto."""
    if override:
        return Path(override).expanduser().resolve()
    p = Path(cfg["paths"]["output_dir"]).expanduser()
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def parse_device(value: Any) -> int | str | None:
    """'default'/vacío -> None; '3' -> 3; cualquier otro texto -> nombre (subcadena)."""
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
    """Siguiente índice libre: máximo existente + 1 (empieza en 1)."""
    if not category_dir.is_dir():
        return 1
    rx = _filename_regex(category)
    indices = [int(m.group(1)) for p in category_dir.iterdir() if (m := rx.match(p.name))]
    return max(indices, default=0) + 1


def save_take(out_dir: Path, category: str, audio: np.ndarray) -> Path:
    """Guarda una toma (ya en formato objetivo) con el siguiente nombre libre."""
    category_dir = out_dir / category
    category_dir.mkdir(parents=True, exist_ok=True)
    idx = next_index(category_dir, category)
    for _ in range(1000):
        path = category_dir / f"{category}_{idx:04d}.wav"
        try:
            return save_wav(path, audio, TARGET_SR)  # creación exclusiva
        except FileExistsError:
            idx += 1
    raise RuntimeError(f"No se encontró un nombre libre en {category_dir}")


def check_take(stats, min_seconds: float) -> list[str]:
    """Avisos sobre la calidad de una toma. No descarta nada."""
    warnings: list[str] = []
    if stats.duration_s < min_seconds:
        warnings.append(f"muy corta ({stats.duration_s:.2f} s; mínimo recomendado {min_seconds:g} s)")
    if stats.peak_dbfs < SILENCE_PEAK_DBFS:
        warnings.append(f"casi silencio (pico {stats.peak_dbfs:.0f} dBFS): ¿micro silenciado o mal elegido?")
    if stats.clipped_fraction > CLIPPING_FRACTION:
        warnings.append("saturación (clipping): baja la ganancia del micro o aléjate")
    return warnings


# --------------------------------------------------------------------------
# Audio en vivo (requiere sounddevice)
# --------------------------------------------------------------------------
def import_sounddevice():
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise SystemExit(
            f"No se pudo importar 'sounddevice' ({exc}).\n"
            'Instálalo con:  pip install -e ".[record]"\n'
            "En Linux también hace falta PortAudio:  sudo apt install libportaudio2"
        ) from exc
    return sd


def resolve_input_rate(sd, device) -> int:
    """16 kHz si el dispositivo lo admite; si no, su frecuencia nativa."""
    try:
        sd.check_input_settings(device=device, channels=1, dtype="int16", samplerate=TARGET_SR)
        return TARGET_SR
    except sd.PortAudioError:
        info = sd.query_devices(device, "input")
        return int(info["default_samplerate"])


class MicCapture:
    """Captura mono int16 desde el micro hasta que se llame a ``stop()``."""

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

    # El callback corre en el hilo de audio de PortAudio: debe ser corto.
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
        self._peak = max(self._peak, float(np.max(np.abs(block.astype(np.int32)))) / 32768.0)

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
        self._thread = threading.Thread(target=self._show_progress, daemon=True)
        self._thread.start()

    def _show_progress(self) -> None:
        while not self._stop_evt.is_set():
            peak, self._peak = self._peak, 0.0
            db = 20.0 * math.log10(max(peak, 1e-6))
            filled = int(max(0.0, min(1.0, (db + 60.0) / 60.0)) * self.BAR_WIDTH)
            bar = "#" * filled + "-" * (self.BAR_WIDTH - filled)
            elapsed = self._frames / self.rate
            tail = "LIMITE: pulsa ENTER" if self._limit_hit else "ENTER = terminar"
            sys.stdout.write(
                f"\r   [REC] {elapsed:6.1f}s/{self.max_seconds:g}s [{bar}] {db:6.1f} dBFS  {tail}   "
            )
            sys.stdout.flush()
            self._stop_evt.wait(0.1)

    def stop(self) -> np.ndarray:
        """Detiene la captura y devuelve el audio a la frecuencia de captura."""
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
    """Reproduce audio y espera. Ctrl+C corta la reproducción sin salir del programa."""
    try:
        try:
            sd.play(audio, sr, device=device)
        except sd.PortAudioError:
            info = sd.query_devices(device, "output")
            native = int(info["default_samplerate"])
            sd.play(resample(audio, sr, native), native, device=device)
        sd.wait()
    except KeyboardInterrupt:
        sd.stop()


# --------------------------------------------------------------------------
# Sesión interactiva
# --------------------------------------------------------------------------
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
    except Exception as exc:  # noqa: BLE001 - PortAudioError, ValueError...
        print(f"No se pudo acceder al micrófono: {exc}")
        print("Usa --list-devices para ver los dispositivos disponibles.")
        return 1

    category_dir = out_dir / category
    category_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nCategoría : {category}")
    print(f"Carpeta   : {category_dir}")
    print(f"Micrófono : {dev_name}")
    if rate != TARGET_SR:
        print(f"            (captura a {rate} Hz; se convierte a {TARGET_SR} Hz al guardar)")
    print(f"Formato   : WAV PCM16 mono {TARGET_SR} Hz | duración máx. por toma: {max_s:g} s")

    done = 0
    last_audio: np.ndarray | None = None

    while done < takes:
        total = count_existing(category_dir, category)
        print(f"\n[{done + 1}/{takes}] {category}   (archivos en carpeta: {total})")
        print("   ENTER = empezar   Q = salir")
        if wait_key({"enter", "q"}) == "q":
            break

        capture = MicCapture(sd, in_dev, rate, max_s)
        try:
            capture.start()
        except Exception as exc:  # noqa: BLE001
            print(f"No se pudo iniciar la grabación: {exc}")
            return 1
        try:
            wait_key({"enter"})
        finally:
            raw = capture.stop()  # también se ejecuta con Ctrl+C

        if raw.size == 0:
            print("   No se capturó audio; inténtalo de nuevo.")
            continue

        audio = to_target_format(raw, rate)
        stats = audio_stats(audio, TARGET_SR)
        path = save_take(out_dir, category, audio)
        done += 1
        last_audio = audio

        print(f"   guardado {path.name}  ({stats.duration_s:.1f} s, pico {stats.peak_dbfs:.1f} dBFS)")
        for warning in check_take(stats, min_s):
            print(f"   ! {warning}  -> pulsa R para repetir")

        quit_now = False
        while True:
            next_label = "terminar" if done >= takes else "siguiente"
            print(f"   [ENTER] {next_label}  [R] repetir  [P] reproducir  [Q] salir")
            key = wait_key({"enter", "r", "p", "q"})
            if key == "enter":
                break
            if key == "p":
                play_audio(sd, last_audio, TARGET_SR, out_dev)
            elif key == "r":
                path.unlink(missing_ok=True)  # solo la toma que creó esta sesión
                print(f"   borrado {path.name}; se repite la toma")
                done -= 1
                last_audio = None
                break
            elif key == "q":
                quit_now = True
                break
        if quit_now:
            break

    print(f"\nSesión terminada: {done} toma(s) guardada(s) en {category_dir}")
    return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def ask_category() -> str:
    print("¿Qué quieres grabar?")
    for i, name in enumerate(CATEGORIES, 1):
        print(f"  {i}) {name}")
    while True:
        raw = input("Elige 1-3 o escribe el nombre: ").strip().lower()
        if raw.isdigit() and 1 <= int(raw) <= len(CATEGORIES):
            return CATEGORIES[int(raw) - 1]
        if raw in CATEGORIES:
            return raw
        print("Opción no válida.")


def ask_takes(default: int) -> int:
    while True:
        raw = input(f"¿Cuántas tomas? [{default}]: ").strip()
        if not raw:
            return default
        if raw.isdigit() and int(raw) > 0:
            return int(raw)
        print("Introduce un número entero mayor que 0.")


def positive_int(value: str) -> int:
    n = int(value)
    if n <= 0:
        raise argparse.ArgumentTypeError("debe ser mayor que 0")
    return n


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m record.recorder",
        description="Graba el dataset de WakeForge desde el terminal.",
    )
    p.add_argument("-c", "--category", choices=CATEGORIES, help="tipo de audio a grabar")
    p.add_argument("-n", "--takes", type=positive_int, help="número de tomas de la sesión")
    p.add_argument("--config", help=f"ruta a un config.toml (por defecto {CONFIG_PATH})")
    p.add_argument("--output-dir", help="carpeta de salida (por defecto, la de config.toml)")
    p.add_argument("--input-device", help="índice o nombre del micrófono")
    p.add_argument("--output-device", help="índice o nombre del altavoz")
    p.add_argument("--list-devices", action="store_true", help="lista los dispositivos de audio y sale")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    sd = import_sounddevice()

    if args.list_devices:
        print(sd.query_devices())
        return 0

    in_dev = parse_device(args.input_device if args.input_device is not None else cfg["device"]["input"])
    out_dev = parse_device(args.output_device if args.output_device is not None else cfg["device"]["output"])
    out_dir = resolve_output_dir(cfg, args.output_dir)

    try:
        category = args.category or ask_category()
        takes = args.takes or ask_takes(int(cfg[category].get("default_takes", 50)))
        return run_session(sd, cfg, category, takes, out_dir, in_dev, out_dev)
    except (KeyboardInterrupt, EOFError):
        print("\nInterrumpido. Las tomas ya guardadas se conservan.")
        return 130
    finally:
        sd.stop()


if __name__ == "__main__":
    sys.exit(main())