"""wakeforge: entrena tu propio wake word en casa."""
from .config import Config, load_config

__all__ = ["Config", "load_config", "train"]


def train(cfg: Config):
    """Entrena con las grabaciones de cfg.project.recordings_dir y exporta el ONNX."""
    from .train import train as _train  # import perezoso: torch solo se carga al entrenar
    return _train(cfg)