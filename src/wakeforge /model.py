"""Red pequeña (DS-CNN temporal con residuales) y wrapper para exportar a ONNX."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelCfg


class DSBlock(nn.Module):
    """Conv 1D depthwise-separable + residual."""

    def __init__(self, c_in: int, c_out: int, k: int, dilation: int, p: float):
        super().__init__()
        pad = (k - 1) // 2 * dilation
        self.dw = nn.Conv1d(c_in, c_in, k, padding=pad, dilation=dilation, groups=c_in, bias=False)
        self.pw = nn.Conv1d(c_in, c_out, 1, bias=False)
        self.bn = nn.BatchNorm1d(c_out)
        self.drop = nn.Dropout(p)
        self.skip = nn.Identity() if c_in == c_out else nn.Sequential(
            nn.Conv1d(c_in, c_out, 1, bias=False), nn.BatchNorm1d(c_out))

    def forward(self, x):
        y = self.drop(F.relu(self.bn(self.pw(self.dw(x)))))
        return F.relu(y + self.skip(x))


class WakeWordNet(nn.Module):
    """log-mel (B, n_mels, T) -> logit (B,). Funciona con cualquier T."""

    def __init__(self, n_mels: int, cfg: ModelCfg):
        super().__init__()
        w = cfg.width
        self.in_bn = nn.BatchNorm1d(n_mels)
        self.stem = nn.Sequential(nn.Conv1d(n_mels, w, 5, padding=2, bias=False),
                                  nn.BatchNorm1d(w), nn.ReLU())
        self.blocks = nn.Sequential(
            *[DSBlock(w, w, cfg.kernel_size, int(d), cfg.dropout) for d in cfg.dilations])
        self.fc = nn.Linear(2 * w, 1)

    def forward(self, x):
        x = self.blocks(self.stem(self.in_bn(x)))
        x = torch.cat([x.mean(-1), x.amax(-1)], 1)  # el max captura el pico de la palabra
        return self.fc(x).squeeze(-1)


class ExportModel(nn.Module):
    """audio crudo (B, samples) -> probabilidad (B,). Es lo que se guarda en el .onnx."""

    def __init__(self, frontend: nn.Module, net: nn.Module):
        super().__init__()
        self.frontend, self.net = frontend, net

    def forward(self, audio):
        return torch.sigmoid(self.net(self.frontend(audio)))