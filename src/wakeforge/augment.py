from __future__ import annotations

import torch
import torch.nn as nn
import torchaudio

from .config import AugmentationCfg


def mix_snr(
    signal: torch.Tensor,
    noise: torch.Tensor,
    snr_db: float,
) -> torch.Tensor:
    ps = signal.pow(2).mean().clamp_min(1e-8)
    pn = noise.pow(2).mean().clamp_min(1e-8)
    return signal + torch.sqrt(
        ps / (pn * 10 ** (snr_db / 10))
    ) * noise


def change_speed(
    wav: torch.Tensor,
    factor: float,
    sr: int,
) -> torch.Tensor:
    if abs(factor - 1.0) < 1e-3:
        return wav
    return torchaudio.functional.resample(
        wav,
        int(round(sr * factor)),
        sr,
    )


def build_specaug(cfg: AugmentationCfg) -> nn.Module | None:
    layers = []

    if cfg.freq_mask > 0:
        layers.append(
            torchaudio.transforms.FrequencyMasking(
                cfg.freq_mask,
                iid_masks=True,
            )
        )

    if cfg.time_mask > 0:
        layers.append(
            torchaudio.transforms.TimeMasking(
                cfg.time_mask,
                iid_masks=True,
            )
        )

    return nn.Sequential(*layers) if layers else None