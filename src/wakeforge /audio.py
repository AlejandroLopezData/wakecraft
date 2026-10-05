"""Carga de audio, remuestreo y recorte de silencios."""
from __future__ import annotations

from pathlib import Path

import torch

AUDIO_EXTS = {".wav", ".flac", ".ogg", ".mp3", ".m4a"}


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