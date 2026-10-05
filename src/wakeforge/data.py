"""Carga de my_recordings/, split por archivo (sin fugas) y Dataset de entrenamiento."""
from __future__ import annotations

import random
import warnings
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .audio import list_audio, load_audio, trim_silence
from .augment import change_speed, mix_snr
from .config import AugmentationCfg, Config
from .window import choose_window

Item = tuple[Path, torch.Tensor]


@dataclass
class PreparedData:
    train_ds: "WakeDataset"
    val_ds: "WakeDataset"
    window_samples: int
    info: dict
    split: dict


def _load_folder(folder: Path, sr: int, trim: bool = False, top_db: float = 30.0) -> list[Item]:
    items = []
    for p in list_audio(folder):
        try:
            w = load_audio(p, sr)
        except RuntimeError as e:
            warnings.warn(str(e))
            continue
        if trim:
            w = trim_silence(w, sr, top_db)
        if w.numel() < sr * 0.1 or w.abs().max() < 1e-4:
            warnings.warn(f"{p.name}: audio vacío, demasiado corto o en silencio. Se ignora.")
            continue
        items.append((p, w))
    return items


def _split(items: list[Item], frac: float, seed: int):
    """Split por ARCHIVO. Con un único archivo se parte por tiempo (último tramo a validación)."""
    if len(items) >= 2:
        order = list(range(len(items)))
        random.Random(seed).shuffle(order)
        n_val = min(max(1, round(len(items) * frac)), len(items) - 1)
        return ([items[i] for i in order[n_val:]], [items[i] for i in order[:n_val]], "by_file")
    if len(items) == 1:
        p, w = items[0]
        cut = int(len(w) * (1 - frac))
        return [(p, w[:cut])], [(p, w[cut:])], "by_time"
    return [], [], "none"


def _chunks(w: torch.Tensor, window: int, stride: int) -> list[torch.Tensor]:
    if w.numel() <= window:
        return [w]
    starts = list(range(0, w.numel() - window + 1, stride))
    if starts[-1] != w.numel() - window:
        starts.append(w.numel() - window)
    return [w[s:s + window] for s in starts]


class WakeDataset(Dataset):
    """Ejemplos (audio de longitud fija, etiqueta). El ruido se mezcla al vuelo."""

    def __init__(self, items: list[tuple[torch.Tensor, int]], noise: list[torch.Tensor],
                 window: int, sr: int, aug: AugmentationCfg, train: bool):
        self.items, self.noise, self.window, self.sr = items, noise, window, sr
        self.aug, self.train = aug, train
        self.labels = [y for _, y in items]

    def __len__(self):
        return len(self.items)

    def _fit(self, w: torch.Tensor, rng) -> torch.Tensor:
        n, W = w.numel(), self.window
        if n < W:
            pad = W - n
            left = rng.randint(0, pad) if self.train else pad // 2
            return F.pad(w, (left, pad - left))
        if n > W:
            s = rng.randint(0, n - W) if self.train else (n - W) // 2
            return w[s:s + W]
        return w

    def __getitem__(self, i):
        a = self.aug
        rng = random if self.train else random.Random(i)  # validación determinista
        w, y = self.items[i]
        if self.train and rng.random() < a.speed_prob:
            w = change_speed(w, rng.choice(a.speed_factors), self.sr)
        w = self._fit(w, rng)
        if self.train:
            w = w * 10 ** (rng.uniform(-a.gain_db, a.gain_db) / 20)
        p_noise = a.noise_prob if self.train else 0.5
        if self.noise and rng.random() < p_noise:
            n = self._fit(rng.choice(self.noise), rng)
            w = mix_snr(w, n, rng.uniform(a.snr_min_db, a.snr_max_db))
        return w.clamp(-1, 1), torch.tensor(float(y))


def prepare_data(cfg: Config) -> PreparedData:
    root, sr, t = Path(cfg.project.recordings_dir), cfg.audio.sample_rate, cfg.training

    pos = _load_folder(root / "wake_word", sr, cfg.audio.trim_silence, cfg.audio.trim_top_db)
    neg = _load_folder(root / "negatives", sr)
    noise = _load_folder(root / "noise", sr)

    if len(pos) < 4:
        raise ValueError(f"Hacen falta al menos 4 grabaciones en {root / 'wake_word'} "
                         f"(encontradas: {len(pos)}). Lo recomendable son 30-50.")
    if not neg:
        raise ValueError(f"No hay audio en {root / 'negatives'}. Sin negativos el modelo se "
                         "activaría con cualquier cosa: graba habla normal y palabras parecidas.")
    if len(pos) < 20:
        warnings.warn(f"Solo {len(pos)} grabaciones de la wake word: el modelo y las métricas serán poco fiables.")
    if sum(w.numel() for _, w in neg) / sr < 60:
        warnings.warn("Menos de 60 s de negativos: espera muchos falsos positivos.")
    if not noise:
        warnings.warn("Sin ruido en my_recordings/noise: el modelo será frágil con ruido ambiente.")

    window_s, wstats = choose_window([w.numel() / sr for _, w in pos], cfg.audio)
    window = int(round(window_s * sr))
    if wstats["inconsistent_durations"]:
        warnings.warn("Las grabaciones de la wake word varían mucho de duración; revisa que todas digan lo mismo.")
    print(f"[data] ventana: {window_s:.2f} s ({wstats['window_mode']})")

    p_tr, p_va, p_mode = _split(pos, t.val_fraction, t.seed)
    n_tr, n_va, n_mode = _split(neg, t.val_fraction, t.seed)
    z_tr, z_va, z_mode = _split(noise, t.val_fraction, t.seed)

    stride_tr = max(1, int(window * t.negative_stride))

    def build(p, n, z, stride):
        items = [(w, 1) for _, w in p]
        for _, w in list(n) + list(z):  # el ruido solo también cuenta como negativo
            items += [(c, 0) for c in _chunks(w, window, stride)]
        return items

    train_items = build(p_tr, n_tr, z_tr, stride_tr)
    val_items = build(p_va, n_va, z_va, window)
    mk = lambda items, z, train: WakeDataset(items, [w for _, w in z], window, sr, cfg.augmentation, train)

    split = {
        "modes": {"wake_word": p_mode, "negatives": n_mode, "noise": z_mode},
        "train": {k: [p.name for p, _ in v] for k, v in
                  (("wake_word", p_tr), ("negatives", n_tr), ("noise", z_tr))},
        "val": {k: [p.name for p, _ in v] for k, v in
                (("wake_word", p_va), ("negatives", n_va), ("noise", z_va))},
    }
    count = lambda items, y: sum(1 for _, l in items if l == y)
    info = {
        **wstats,
        "files": {"wake_word": len(pos), "negatives": len(neg), "noise": len(noise)},
        "train_examples": {"positive": count(train_items, 1), "negative": count(train_items, 0)},
        "val_examples": {"positive": count(val_items, 1), "negative": count(val_items, 0)},
        "negatives_hours": sum(w.numel() for _, w in neg) / sr / 3600,
    }
    return PreparedData(mk(train_items, z_tr, True), mk(val_items, z_va, False), window, info, split)