"""Entrenamiento y exportación: my_recordings/ -> outputs/<name>/model.onnx + metadata.json."""
from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from .augment import build_specaug
from .config import Config, TrainingCfg
from .data import prepare_data
from .export import export_onnx
from .features import LogMelFrontend
from .model import ExportModel, WakeWordNet


class EMA:
    """Media móvil exponencial de pesos (suele generalizar mejor)."""

    def __init__(self, model: nn.Module, decay: float):
        self.m = copy.deepcopy(model).eval()
        self.decay, self.step = decay, 0
        for p in self.m.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        self.step += 1
        d = min(self.decay, (1 + self.step) / (10 + self.step))
        for e, p in zip(self.m.state_dict().values(), model.state_dict().values()):
            if e.dtype.is_floating_point:
                e.mul_(d).add_(p, alpha=1 - d)
            else:
                e.copy_(p)


@torch.no_grad()
def predict(net, frontend, loader, device):
    net.eval()
    logits, ys = [], []
    for w, y in loader:
        logits.append(net(frontend(w.to(device))).float().cpu())
        ys.append(y)
    return torch.cat(logits), torch.cat(ys)


def evaluate_split(logits: torch.Tensor, y: torch.Tensor, t: TrainingCfg) -> dict:
    """Elige el umbral para un FPR máximo (por ventana) y mide el FRR resultante."""
    p = torch.sigmoid(logits)
    pos, neg = p[y == 1], p[y == 0].sort(descending=True).values
    k = max(int(len(neg) * t.max_fpr), 1)
    thr = max(neg[k - 1].item(), t.min_threshold)
    return {
        "loss": F.binary_cross_entropy_with_logits(logits, y).item(),
        "threshold": thr,
        "frr": (pos <= thr).float().mean().item(),
        "fpr": (neg > thr).float().mean().item(),
        "n_val_pos": int(len(pos)),
        "n_val_neg": int(len(neg)),
    }


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def train(cfg: Config) -> Path:
    t = cfg.training
    random.seed(t.seed)
    torch.manual_seed(t.seed)
    device = _device(t.device)
    out = Path(cfg.project.output_dir) / cfg.project.name
    out.mkdir(parents=True, exist_ok=True)

    data = prepare_data(cfg)
    train_ds, val_ds = data.train_ds, data.val_ds

    # sampler balanceado, con reemplazo: funciona aunque haya muy pocos datos
    lab = torch.tensor(train_ds.labels)
    npos, nneg = int((lab == 1).sum()), int((lab == 0).sum())
    if npos == 0 or nneg == 0:
        raise ValueError("El conjunto de entrenamiento necesita positivos y negativos.")
    weights = torch.where(lab == 1, t.positive_ratio / npos, (1 - t.positive_ratio) / nneg)
    sampler = WeightedRandomSampler(weights, t.steps * t.batch_size, replacement=True)
    kw = dict(num_workers=t.num_workers, pin_memory=device.type == "cuda",
              persistent_workers=t.num_workers > 0)
    train_loader = DataLoader(train_ds, t.batch_size, sampler=sampler, drop_last=True, **kw)
    val_loader = DataLoader(val_ds, 128, shuffle=False, **kw)

    frontend = LogMelFrontend(cfg.audio.sample_rate, cfg.features).to(device).eval()
    net = WakeWordNet(cfg.features.n_mels, cfg.model).to(device)
    specaug = build_specaug(cfg.augmentation)
    specaug = specaug.to(device) if specaug is not None else None
    ema = EMA(net, t.ema_decay)
    print(f"[train] {sum(p.numel() for p in net.parameters()) / 1e3:.1f}K parámetros | "
          f"device={device} | pasos={t.steps} | train {npos} pos / {nneg} neg")

    opt = torch.optim.AdamW(net.parameters(), lr=t.lr, weight_decay=t.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, t.lr, total_steps=t.steps, pct_start=0.1)
    ls = t.label_smoothing

    best_key, best_metrics, running = (2.0, 1e9), None, []
    for step, (wav, y) in enumerate(train_loader, start=1):
        net.train()
        wav, y = wav.to(device), y.to(device)
        with torch.no_grad():
            x = frontend(wav)
            if specaug is not None:
                x = specaug(x)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = net(x)
        loss = F.binary_cross_entropy_with_logits(logits.float(), y * (1 - 2 * ls) + ls)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), t.grad_clip)
        opt.step()
        sched.step()
        ema.update(net)
        running.append(loss.item())

        if step % t.eval_every == 0 or step == t.steps:
            lg, yv = predict(ema.m, frontend, val_loader, device)
            m = evaluate_split(lg, yv, t)
            print(f"[train] paso {step:5d}/{t.steps} loss {sum(running) / len(running):.4f} | "
                  f"val loss {m['loss']:.4f} FRR {m['frr']:.3f} FPR {m['fpr']:.4f} thr {m['threshold']:.3f}")
            running.clear()
            key = (m["frr"], m["loss"])
            if key < best_key:
                best_key, best_metrics = key, {**m, "step": step}
                torch.save(ema.m.state_dict(), out / "checkpoint.pt")

    # ---- exportación del mejor modelo
    ema.m.load_state_dict(torch.load(out / "checkpoint.pt", map_location="cpu"))
    export_model = ExportModel(frontend.cpu(), ema.m.cpu())
    onnx_path = out / cfg.export.filename
    diff = export_onnx(export_model, data.window_samples, onnx_path, cfg.export.opset, cfg.export.verify)

    sr = cfg.audio.sample_rate
    metadata = {
        "name": cfg.project.name,
        "model_file": cfg.export.filename,
        "input": {"name": "audio", "shape": ["batch", "samples"], "dtype": "float32",
                  "range": [-1.0, 1.0], "sample_rate": sr},
        "output": {"name": "score", "meaning": "probabilidad de wake word (0-1)"},
        "window_seconds": data.window_samples / sr,
        "window_samples": data.window_samples,
        "suggested_hop_seconds": 0.1,
        "threshold": best_metrics["threshold"],
        "validation": {**best_metrics,
                       "warning": "Pocos ejemplos de validación: tómalo como orientativo."
                       if best_metrics["n_val_pos"] < 20 else None},
        "onnx_max_diff": diff,
        "data": data.info,
        "config": cfg.to_dict(),
        "torch_version": torch.__version__,
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    (out / "split.json").write_text(json.dumps(data.split, indent=2))
    print(f"[train] listo -> {onnx_path}")
    return onnx_path