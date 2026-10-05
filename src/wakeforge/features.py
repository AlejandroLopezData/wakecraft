"""Frontend log-mel implementado con convoluciones.

Se usa tanto en entrenamiento como DENTRO del ONNX exportado, así el modelo final recibe
audio crudo y no hay que reimplementar el preprocesado al usarlo. No usa torch.stft, que
da problemas al exportar (números complejos, padding).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from .config import FeaturesCfg


class LogMelFrontend(nn.Module):
    """audio (B, samples) en [-1, 1] -> log-mel (B, n_mels, frames)."""

    def __init__(self, sample_rate: int, cfg: FeaturesCfg):
        super().__init__()
        n_fft, win, hop = cfg.n_fft, cfg.win_length, cfg.hop_length
        self.n_fft, self.hop = n_fft, hop

        window = torch.hann_window(win, periodic=True)
        left = (n_fft - win) // 2
        window = F.pad(window, (left, n_fft - win - left))

        k = torch.arange(n_fft // 2 + 1, dtype=torch.float32)[:, None]
        n = torch.arange(n_fft, dtype=torch.float32)[None, :]
        ang = 2 * math.pi * k * n / n_fft
        basis = torch.cat([torch.cos(ang) * window, -torch.sin(ang) * window], 0).unsqueeze(1)
        fb = torchaudio.functional.melscale_fbanks(
            n_fft // 2 + 1, cfg.f_min, cfg.f_max, cfg.n_mels, sample_rate, norm=None, mel_scale="htk")

        self.register_buffer("basis", basis)        # (2*(n_fft/2+1), 1, n_fft)
        self.register_buffer("fb_t", fb.t().contiguous())  # (n_mels, n_fft/2+1)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        x = F.pad(audio.unsqueeze(1), (self.n_fft // 2, self.n_fft // 2), mode="reflect")
        spec = F.conv1d(x, self.basis, stride=self.hop)
        re, im = spec.chunk(2, dim=1)
        power = re * re + im * im
        return torch.log(torch.matmul(self.fb_t, power) + 1e-6)