"""Prueba de humo: genera audio sintético en una carpeta temporal, entrena pocos pasos y comprueba el ONNX.

    python scripts/smoke_test.py
"""
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from wakeforge import load_config, train

SR = 16000
rng = np.random.default_rng(0)


def tone(freqs, dur):
    t = np.arange(int(dur * SR)) / SR
    return sum(np.sin(2 * np.pi * f * t) for f in freqs) * 0.3 * np.hanning(t.size)


with tempfile.TemporaryDirectory() as tmp:
    rec = Path(tmp) / "my_recordings"
    for sub in ("wake_word", "negatives", "noise"):
        (rec / sub).mkdir(parents=True)
    for i in range(12):   # "palabra" = dos tonos fijos con jitter
        sf.write(rec / "wake_word" / f"w{i}.wav", tone([500 + rng.integers(-20, 20), 1200], 0.8), SR)
    for i in range(6):    # negativos: otras frecuencias, 10 s
        sf.write(rec / "negatives" / f"n{i}.wav", tone([300 + 90 * i, 2000], 10), SR)
    for i in range(3):    # ruido
        sf.write(rec / "noise" / f"z{i}.wav", rng.normal(0, 0.05, SR * 8), SR)

    cfg = load_config(None, [
        f"project.recordings_dir='{rec}'", f"project.output_dir='{Path(tmp) / 'out'}'",
        "project.name='smoke'", "training.steps=60", "training.eval_every=20", "training.batch_size=16",
    ])
    onnx = train(cfg)
    assert onnx.exists() and (onnx.parent / "metadata.json").exists()
    print("SMOKE TEST OK")