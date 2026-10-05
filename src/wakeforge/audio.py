"""Carga de audio, remuestreo y recorte de silencios."""
from __future__ import annotations

from pathlib import Path

import torch

AUDIO_EXTS = {".wav", ".flac", ".ogg", ".mp3", ".m4a"}

import wave
from dataclasses import dataclass
from math import gcd, log10
from pathlib import Path

import numpy as np

TARGET_SR = 16_000
TARGET_CHANNELS = 1
TARGET_DTYPE = "int16"
SAMPLE_WIDTH_BYTES = 2

_INT16_MAX = 32767
_FLOOR_DBFS = -120.0


def list_audio(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in AUDIO_EXTS)


def load_audio(path: str | Path, sample_rate: int) -> torch.Tensor:
    """Devuelve un tensor 1D float32 mono a `sample_rate`. Lanza RuntimeError si no se puede leer."""
    import torchaudio

    try:
        import soundfile as sf

        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        wav = torch.from_numpy(data.mean(1))
    except Exception:
        try:  # fallback (mp3/m4a según backend instalado)
            wav, sr = torchaudio.load(str(path))
            wav = wav.mean(0)
        except Exception as e:
            raise RuntimeError(
                f"No se pudo leer {path}. Convierte el archivo a .wav (p. ej. con ffmpeg)."
            ) from e
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    return wav.contiguous()


def trim_silence(wav: torch.Tensor, sr: int, top_db: float = 30.0, pad_s: float = 0.05) -> torch.Tensor:
    """Quita silencio inicial/final (energía por frames de 20 ms respecto al pico)."""
    frame = int(0.02 * sr)
    n = wav.numel() // frame
    if n < 2:
        return wav
    rms = wav[: n * frame].view(n, frame).pow(2).mean(1).sqrt().clamp_min(1e-8)
    db = 20 * torch.log10(rms)
    keep = (db > db.max() - top_db).nonzero().flatten()
    pad = int(pad_s * sr)
    start = max(int(keep[0]) * frame - pad, 0)
    end = min((int(keep[-1]) + 1) * frame + pad, wav.numel())
    return wav[start:end]

"""Utilidades de audio reutilizables de WakeForge.

Fuente única de verdad del formato objetivo del dataset:
WAV, PCM 16-bit, mono, 16 kHz.

Convención de arrays: forma ``(frames,)`` para mono o ``(frames, canales)``
para multicanal (la misma que usa ``sounddevice``).

Este módulo no toca hardware: solo depende de numpy (y de scipy, si está
instalado, para un remuestreo de mejor calidad).
"""




# --------------------------------------------------------------------------
# Conversiones
# --------------------------------------------------------------------------
def to_mono(audio: np.ndarray) -> np.ndarray:
    """Devuelve el audio como 1-D. Si es multicanal, promedia los canales."""
    arr = np.asarray(audio)
    if arr.ndim == 1:
        return arr
    if arr.ndim != 2:
        raise ValueError(f"Se esperaba un array 1-D o 2-D, no {arr.ndim}-D")
    if arr.shape[1] == 1:
        return arr[:, 0]
    mixed = arr.astype(np.float64).mean(axis=1)
    if np.issubdtype(arr.dtype, np.integer):
        return np.rint(mixed).astype(arr.dtype)
    return mixed.astype(arr.dtype)


def to_int16(audio: np.ndarray) -> np.ndarray:
    """Convierte a int16. Los floats se interpretan en el rango [-1, 1]."""
    arr = np.asarray(audio)
    if arr.dtype == np.int16:
        return arr
    if np.issubdtype(arr.dtype, np.floating):
        return np.rint(np.clip(arr, -1.0, 1.0) * _INT16_MAX).astype(np.int16)
    raise TypeError(f"dtype no soportado: {arr.dtype} (usa int16 o float)")


def resample(audio: np.ndarray, orig_sr: int, target_sr: int = TARGET_SR) -> np.ndarray:
    """Remuestrea un array mono.

    Usa ``scipy.signal.resample_poly`` si scipy está disponible (con filtro
    antialias). Si no, cae a interpolación lineal, que es aceptable al subir
    la frecuencia pero puede introducir aliasing al bajarla.
    """
    arr = np.asarray(audio)
    if arr.ndim != 1:
        raise ValueError("resample() espera audio mono (1-D)")
    if int(orig_sr) == int(target_sr) or arr.size == 0:
        return arr

    was_int16 = arr.dtype == np.int16
    x = arr.astype(np.float64)

    try:
        from scipy.signal import resample_poly

        g = gcd(int(orig_sr), int(target_sr))
        y = resample_poly(x, int(target_sr) // g, int(orig_sr) // g)
    except ImportError:
        n_out = int(round(len(x) * target_sr / orig_sr))
        t_in = np.arange(len(x)) / orig_sr
        t_out = np.arange(n_out) / target_sr
        y = np.interp(t_out, t_in, x)

    if was_int16:
        return np.clip(np.rint(y), -32768, _INT16_MAX).astype(np.int16)
    return y.astype(np.float32)


def to_target_format(audio: np.ndarray, sr: int) -> np.ndarray:
    """Lleva cualquier audio al formato objetivo: mono, int16, 16 kHz."""
    data = to_int16(to_mono(audio))
    return resample(data, sr, TARGET_SR)


# --------------------------------------------------------------------------
# Entrada / salida WAV
# --------------------------------------------------------------------------
def save_wav(
    path: str | Path,
    audio: np.ndarray,
    sr: int = TARGET_SR,
    *,
    overwrite: bool = False,
) -> Path:
    """Guarda un WAV PCM16 mono.

    Por defecto falla con ``FileExistsError`` si el archivo ya existe (la
    creación es exclusiva, así que no hay carrera entre comprobar y escribir).
    ``sr`` es la frecuencia real del array; esta función NO remuestrea
    (para eso, ``to_target_format``).
    """
    path = Path(path)
    data = to_int16(to_mono(audio))
    path.parent.mkdir(parents=True, exist_ok=True)

    fh = open(path, "wb" if overwrite else "xb")  # FileExistsError sale tal cual
    try:
        with fh:
            with wave.open(fh, "wb") as wf:
                wf.setnchannels(TARGET_CHANNELS)
                wf.setsampwidth(SAMPLE_WIDTH_BYTES)
                wf.setframerate(int(sr))
                wf.writeframes(data.astype("<i2").tobytes())
    except BaseException:
        path.unlink(missing_ok=True)  # no dejar WAV a medias
        raise
    return path


def load_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Lee un WAV PCM16 y devuelve ``(audio_mono_int16, sample_rate)``."""
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    if width != SAMPLE_WIDTH_BYTES:
        raise ValueError(f"{path}: se esperaba PCM de 16 bits, hay {8 * width} bits")
    data = np.frombuffer(raw, dtype="<i2").astype(np.int16)
    if channels > 1:
        data = to_mono(data.reshape(-1, channels))
    return data, sr


# --------------------------------------------------------------------------
# Estadísticas
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class AudioStats:
    duration_s: float
    peak_dbfs: float
    rms_dbfs: float
    clipped_fraction: float  # fracción de muestras en el tope del rango


def _to_dbfs(value: float) -> float:
    if value <= 0:
        return _FLOOR_DBFS
    return max(20.0 * log10(value), _FLOOR_DBFS)


def audio_stats(audio: np.ndarray, sr: int = TARGET_SR) -> AudioStats:
    """Duración, pico y RMS en dBFS, y fracción de muestras saturadas."""
    data = to_int16(to_mono(audio))
    if data.size == 0:
        return AudioStats(0.0, _FLOOR_DBFS, _FLOOR_DBFS, 0.0)
    x = data.astype(np.float64) / 32768.0
    peak = float(np.max(np.abs(x)))
    rms = float(np.sqrt(np.mean(x * x)))
    clipped = float(np.mean(np.abs(data.astype(np.int32)) >= _INT16_MAX))
    return AudioStats(
        duration_s=len(data) / float(sr),
        peak_dbfs=_to_dbfs(peak),
        rms_dbfs=_to_dbfs(rms),
        clipped_fraction=clipped,
    )