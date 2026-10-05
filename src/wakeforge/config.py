"""Carga y validación de config.toml (con valores por defecto y overrides por CLI)."""
from __future__ import annotations

import ast
import sys
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # Python 3.10
    import tomli as tomllib


@dataclass
class ProjectCfg:
    name: str = "my_wakeword"
    recordings_dir: str = "my_recordings"
    output_dir: str = "outputs"


@dataclass
class AudioCfg:
    sample_rate: int = 16000
    window_seconds: float = 0.0
    window_margin: float = 0.4
    window_min: float = 1.0
    window_max: float = 3.0
    window_percentile: float = 90.0
    trim_silence: bool = True
    trim_top_db: float = 30.0


@dataclass
class FeaturesCfg:
    n_mels: int = 40
    n_fft: int = 512
    win_length: int = 400
    hop_length: int = 160
    f_min: float = 20.0
    f_max: float = 7600.0


@dataclass
class ModelCfg:
    width: int = 64
    kernel_size: int = 9
    dilations: list = field(default_factory=lambda: [1, 2, 4, 1, 2, 4])
    dropout: float = 0.1


@dataclass
class TrainingCfg:
    steps: int = 2000
    batch_size: int = 64
    lr: float = 2e-3
    weight_decay: float = 1e-2
    val_fraction: float = 0.2
    positive_ratio: float = 0.33
    label_smoothing: float = 0.01
    ema_decay: float = 0.99
    eval_every: int = 100
    max_fpr: float = 0.005
    min_threshold: float = 0.5
    grad_clip: float = 5.0
    negative_stride: float = 0.5
    seed: int = 0
    num_workers: int = 0
    device: str = "auto"


@dataclass
class AugmentationCfg:
    gain_db: float = 6.0
    noise_prob: float = 0.8
    snr_min_db: float = 0.0
    snr_max_db: float = 20.0
    speed_factors: list = field(default_factory=lambda: [0.9, 0.95, 1.0, 1.05, 1.1])
    speed_prob: float = 0.5
    freq_mask: int = 6
    time_mask: int = 15


@dataclass
class ExportCfg:
    filename: str = "model.onnx"
    opset: int = 17
    verify: bool = True


SECTIONS = {
    "project": ProjectCfg,
    "audio": AudioCfg,
    "features": FeaturesCfg,
    "model": ModelCfg,
    "training": TrainingCfg,
    "augmentation": AugmentationCfg,
    "export": ExportCfg,
}


@dataclass
class Config:
    project: ProjectCfg = field(default_factory=ProjectCfg)
    audio: AudioCfg = field(default_factory=AudioCfg)
    features: FeaturesCfg = field(default_factory=FeaturesCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    training: TrainingCfg = field(default_factory=TrainingCfg)
    augmentation: AugmentationCfg = field(default_factory=AugmentationCfg)
    export: ExportCfg = field(default_factory=ExportCfg)

    def to_dict(self) -> dict:
        return asdict(self)


def _parse_value(text: str):
    low = text.strip().lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def load_config(path: str | Path | None = None, overrides: list[str] | tuple = ()) -> Config:
    """Lee un TOML (opcional) y aplica overrides tipo 'training.steps=500'."""
    raw: dict = {}
    if path is not None:
        with open(path, "rb") as f:
            raw = tomllib.load(f)

    for item in overrides:
        key, sep, val = item.partition("=")
        section, dot, name = key.strip().partition(".")
        if not sep or not dot:
            raise ValueError(f"Override inválido '{item}'. Usa el formato seccion.opcion=valor")
        raw.setdefault(section, {})[name] = _parse_value(val)

    unknown = set(raw) - set(SECTIONS)
    if unknown:
        raise ValueError(f"Secciones desconocidas en la config: {sorted(unknown)}. "
                         f"Válidas: {sorted(SECTIONS)}")

    built = {}
    for name, cls in SECTIONS.items():
        data = raw.get(name, {})
        valid = {f.name for f in fields(cls)}
        bad = set(data) - valid
        if bad:
            raise ValueError(f"[{name}] opciones desconocidas: {sorted(bad)}. Válidas: {sorted(valid)}")
        built[name] = cls(**data)

    cfg = Config(**built)
    validate(cfg)
    return cfg


def validate(cfg: Config) -> None:
    a, f, m, t, g = cfg.audio, cfg.features, cfg.model, cfg.training, cfg.augmentation
    problems = []
    if f.n_fft < f.win_length:
        problems.append("features.n_fft debe ser >= features.win_length")
    if f.f_max > a.sample_rate / 2:
        problems.append("features.f_max no puede superar sample_rate / 2")
    if not 0 < t.val_fraction < 0.5:
        problems.append("training.val_fraction debe estar entre 0 y 0.5")
    if not 0 < t.positive_ratio < 1:
        problems.append("training.positive_ratio debe estar entre 0 y 1")
    if t.steps < 1 or t.batch_size < 1 or t.eval_every < 1:
        problems.append("training.steps, batch_size y eval_every deben ser >= 1")
    if not 0 < t.negative_stride <= 1:
        problems.append("training.negative_stride debe estar en (0, 1]")
    if g.snr_min_db > g.snr_max_db:
        problems.append("augmentation.snr_min_db no puede ser mayor que snr_max_db")
    if not g.speed_factors or any(x <= 0 for x in g.speed_factors):
        problems.append("augmentation.speed_factors debe tener valores > 0")
    if a.window_min > a.window_max:
        problems.append("audio.window_min no puede ser mayor que window_max")
    if not m.dilations:
        problems.append("model.dilations no puede estar vacío")
    if problems:
        raise ValueError("Config inválida:\n  - " + "\n  - ".join(problems))