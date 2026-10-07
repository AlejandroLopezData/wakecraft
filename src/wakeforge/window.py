from __future__ import annotations
import numpy as np
from .config import AudioCfg


def choose_window(
    durations: list[float],
    cfg: AudioCfg,
) -> tuple[float, dict]:
    d = np.asarray(durations, dtype=float)

    stats = {
        "n_recordings": int(d.size),
        "duration_mean_s": float(d.mean()),
        "duration_std_s": float(d.std()),
        "duration_min_s": float(d.min()),
        "duration_max_s": float(d.max()),
    }

    if cfg.window_seconds > 0:
        seconds, mode = cfg.window_seconds, "manual"
    else:
        raw = (
            float(np.percentile(d, cfg.window_percentile))
            + cfg.window_margin
        )

        seconds = min(
            max(raw, cfg.window_min),
            cfg.window_max,
        )

        seconds = round(seconds / 0.05) * 0.05
        mode = "auto"

    stats.update(
        window_mode=mode,
        window_seconds=float(seconds),
    )

    stats["inconsistent_durations"] = bool(
        d.mean() > 0
        and d.std() / d.mean() > 0.35
    )

    return float(seconds), stats