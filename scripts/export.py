"""Exportación a ONNX con verificación de paridad PyTorch vs ONNX Runtime."""
from __future__ import annotations

import copy
from pathlib import Path

import torch

from .model import ExportModel


def export_onnx(model: ExportModel, window_samples: int, path: str | Path,
                opset: int = 17, verify: bool = True) -> float | None:
    """Exporta audio (B, samples) -> score (B,). Devuelve la diferencia máxima torch/onnx."""
    model = copy.deepcopy(model).eval().cpu()
    path = Path(path)
    dummy = torch.randn(1, window_samples) * 0.1
    kw = dict(input_names=["audio"], output_names=["score"], opset_version=opset,
              dynamic_axes={"audio": {0: "batch", 1: "samples"}, "score": {0: "batch"}})
    try:
        torch.onnx.export(model, dummy, str(path), dynamo=False, **kw)
    except TypeError:  # torch antiguo sin argumento `dynamo`
        torch.onnx.export(model, dummy, str(path), **kw)

    if not verify:
        return None
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError:
        print("[export] onnxruntime no instalado: se omite la verificación.")
        return None

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    worst = 0.0
    # se prueba con la ventana y con otra longitud para confirmar el eje dinámico
    for n in (window_samples, int(window_samples * 1.5)):
        x = torch.randn(3, n) * 0.1
        with torch.no_grad():
            ref = model(x).numpy()
        out = sess.run(None, {"audio": x.numpy()})[0]
        worst = max(worst, float(np.abs(ref - out).max()))
    print(f"[export] max |torch - onnx| = {worst:.2e}")
    if worst > 1e-3:
        raise RuntimeError(f"El ONNX no coincide con PyTorch (diferencia {worst:.2e}).")
    return worst