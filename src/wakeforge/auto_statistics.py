"""Informe automático del modelo entrenado.

Genera en outputs/<name>/stats/:
  report.html  informe autocontenido (gráficas + estadísticas + avisos), se abre en el navegador
  stats.json   las mismas estadísticas en formato máquina
  raw.json     datos crudos (pérdida y scores de validación) para regenerar el informe sin reentrenar

Se llama solo al final de train(). Para regenerarlo a mano:
    python -m wakeforge.auto_statistics outputs/my_wakeword

No depende de torch: solo numpy, matplotlib (para las gráficas) y onnxruntime (opcional, latencia).
"""
from __future__ import annotations

import base64
import html
import io
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# ───────────────────────────── utilidades estadísticas ─────────────────────────────


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Intervalo de confianza (95 %) de una proporción k/n. Honesto con muestras pequeñas."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def auc_score(pos: np.ndarray, neg: np.ndarray) -> float | None:
    """Probabilidad de que un positivo puntúe más que un negativo (1.0 = separación perfecta)."""
    if len(pos) == 0 or len(neg) == 0:
        return None
    s = np.sort(neg)
    lt = np.searchsorted(s, pos, "left")
    le = np.searchsorted(s, pos, "right")
    return float((lt + 0.5 * (le - lt)).sum() / (len(pos) * len(neg)))


def benchmark_onnx(path: Path, window_samples: int, hop_s: float, runs: int = 50) -> dict | None:
    """Latencia de inferencia en CPU (1 hilo) para una ventana."""
    try:
        import onnxruntime as ort
    except ImportError:
        return None
    try:
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        x = (np.random.randn(1, window_samples) * 0.1).astype(np.float32)
        for _ in range(5):
            sess.run(None, {"audio": x})
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            sess.run(None, {"audio": x})
            times.append((time.perf_counter() - t0) * 1000)
    except Exception as e:  # el informe no debe romperse por esto
        print(f"[stats] no se pudo medir la latencia: {e}")
        return None
    mean = float(np.mean(times))
    return {
        "mean_ms": mean,
        "p95_ms": float(np.percentile(times, 95)),
        "cpu_load_pct_at_hop": mean / (hop_s * 1000) * 100,
    }


# ───────────────────────────── cálculo de estadísticas ─────────────────────────────


def compute_stats(meta: dict, scores: np.ndarray, labels: np.ndarray, names: list[str],
                  onnx_path: Path | None) -> dict:
    thr = float(meta["threshold"])
    win_s = float(meta["window_seconds"])
    hop_s = float(meta.get("suggested_hop_seconds", 0.1))
    names = names if len(names) == len(scores) else [f"#{i}" for i in range(len(scores))]

    pos, neg = scores[labels == 1], scores[labels == 0]
    tp = int((pos > thr).sum())
    fn = len(pos) - tp
    fp = int((neg > thr).sum())
    tn = len(neg) - fp

    frr = fn / len(pos) if len(pos) else None
    fpr = fp / len(neg) if len(neg) else None
    frr_ci = wilson(fn, len(pos))
    fpr_ci = wilson(fp, len(neg))

    # Falsas activaciones por hora, estimadas con ventanas de validación independientes.
    # En uso real una misma activación dispara varias ventanas seguidas, así que es orientativo.
    per_hour = 3600 / win_s
    neg_hours = len(neg) * win_s / 3600

    ths = np.linspace(0, 1, 201)
    sweep = {
        "thresholds": ths.tolist(),
        "frr": (pos[None, :] <= ths[:, None]).mean(1).tolist() if len(pos) else [],
        "fpr": (neg[None, :] > ths[:, None]).mean(1).tolist() if len(neg) else [],
    }

    def rows(idx, reverse):
        r = sorted(((names[i], float(scores[i])) for i in idx), key=lambda t: t[1], reverse=reverse)
        return [{"file": n, "score": s} for n, s in r]

    pos_idx, neg_idx = np.where(labels == 1)[0], np.where(labels == 0)[0]
    missed = [r for r in rows(pos_idx, False) if r["score"] <= thr]
    false_alarms = [r for r in rows(neg_idx, True) if r["score"] > thr]

    stats = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "threshold": thr,
        "window_seconds": win_s,
        "counts": {"val_positives": len(pos), "val_negatives": len(neg),
                   "tp": tp, "fn": fn, "fp": fp, "tn": tn},
        "recall": tp / len(pos) if len(pos) else None,
        "frr": frr, "frr_ci95": frr_ci,
        "fpr": fpr, "fpr_ci95": fpr_ci,
        "auc": auc_score(pos, neg),
        "score_median": {"positives": float(np.median(pos)) if len(pos) else None,
                         "negatives": float(np.median(neg)) if len(neg) else None},
        "val_negative_minutes": neg_hours * 60,
        "false_activations_per_hour": (fp / neg_hours) if neg_hours > 0 else None,
        "false_activations_per_hour_upper95": fpr_ci[1] * per_hour if len(neg) else None,
        "missed_positives": missed,
        "false_alarms": false_alarms,
        "weakest_positives": rows(pos_idx, False)[:5],
        "strongest_negatives": rows(neg_idx, True)[:5],
        "sweep": sweep,
        "latency": benchmark_onnx(onnx_path, int(meta["window_samples"]), hop_s)
        if onnx_path and Path(onnx_path).exists() else None,
        "onnx_size_kb": Path(onnx_path).stat().st_size / 1024
        if onnx_path and Path(onnx_path).exists() else None,
        "n_parameters": meta.get("n_parameters"),
    }
    return stats


def make_advice(meta: dict, s: dict, losses: list[float]) -> list[dict]:
    """Avisos y recomendaciones en lenguaje llano."""
    out = []
    d = meta.get("data", {})
    c = s["counts"]
    add = lambda level, text: out.append({"level": level, "text": text})

    n_rec = d.get("n_recordings", 0)
    if n_rec < 20:
        add("warn", f"Solo hay {n_rec} grabaciones de la wake word. Intenta llegar a 30-50, con distintos "
                    "volúmenes, distancias y tonos de voz.")
    if d.get("negatives_hours", 0) < 0.1:
        add("warn", f"Solo hay {d.get('negatives_hours', 0) * 60:.1f} min de audio negativo. Es lo que más "
                    "reduce los falsos positivos: añade 10-15 min de habla normal, tele, música y palabras parecidas.")
    if d.get("files", {}).get("noise", 0) == 0:
        add("warn", "No hay ruido en my_recordings/noise: el modelo será frágil con ruido ambiente.")
    if d.get("inconsistent_durations"):
        add("warn", "Las grabaciones de la palabra varían mucho de duración. Revisa que todas digan lo mismo.")
    if c["val_positives"] < 20:
        add("warn", f"La validación solo tiene {c['val_positives']} positivos y {c['val_negatives']} negativos: "
                    "los porcentajes saltan mucho con un solo fallo. Mira los intervalos de confianza.")
    if c["fn"] > 0:
        add("info", f"Se perdieron {c['fn']} palabra(s) en validación. Escucha los archivos de la tabla "
                    "'Palabras no detectadas': suelen ser tomas bajas, cortadas o con ruido.")
    if c["fp"] > 0:
        add("info", f"Hay {c['fp']} falsa(s) alarma(s) en validación. Añade audios parecidos a los de la "
                    "tabla 'Falsas alarmas' como negativos.")
    up = s.get("false_activations_per_hour_upper95")
    if up is not None and up > 1:
        add("warn", f"Con {s['val_negative_minutes']:.1f} min de negativos de validación no se puede asegurar "
                    f"pocas falsas activaciones: la tasa real podría llegar a {up:.0f} por hora (95 %). "
                    "Pruébalo con audio largo que no se usara para entrenar.")
    if s.get("auc") is not None and s["auc"] < 0.95:
        add("warn", f"La separación entre palabra y no-palabra es mejorable (AUC {s['auc']:.3f}). "
                    "Faltan datos o variedad.")
    lat = s.get("latency")
    if lat and lat["cpu_load_pct_at_hop"] > 50:
        add("warn", "El modelo tarda más de la mitad del intervalo de escucha en CPU: reduce model.width o dilations.")
    if len(losses) >= 50:
        k = max(5, len(losses) // 10)
        if np.mean(losses[-k:]) > 0.8 * np.mean(losses[:k]):
            add("warn", "La pérdida casi no bajó durante el entrenamiento: prueba más pasos o revisa los datos.")
    out.append({"level": "info", "text": "Estas métricas se miden sobre grabaciones de la misma sesión y micro que el "
                                         "entrenamiento. Para saber cómo irá de verdad, prueba con audio nuevo."})
    if not any(o["level"] == "warn" for o in out):
        out.insert(0, {"level": "ok", "text": "No se detectaron problemas evidentes en los datos ni en las métricas."})
    return out


# ───────────────────────────── gráficas ─────────────────────────────

GREEN, RED, GREY, BLUE = "#16a34a", "#dc2626", "#6b7280", "#2563eb"


def _png(fig) -> str:
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def make_figures(meta: dict, s: dict, losses: list[float], scores: np.ndarray,
                 labels: np.ndarray) -> list[dict]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    thr, d = s["threshold"], meta.get("data", {})
    pos, neg = scores[labels == 1], scores[labels == 0]
    figs = []

    # 1) pérdida de entrenamiento
    if len(losses) > 1:
        fig, ax = plt.subplots(figsize=(6.4, 3.4))
        w = max(1, len(losses) // 30)
        ax.plot(losses, color=GREY, alpha=0.35, lw=1, label="por paso")
        if w > 1:
            ma = np.convolve(losses, np.ones(w) / w, mode="valid")
            ax.plot(np.arange(w - 1, len(losses)), ma, color=BLUE, lw=2, label=f"media ({w} pasos)")
        ax.set_xlabel("paso"); ax.set_ylabel("pérdida"); ax.legend(frameon=False)
        figs.append({"title": "Curva de entrenamiento",
                     "caption": "Debe bajar y estabilizarse. Si sube o se queda plana, algo va mal con los datos o el learning rate.",
                     "img": _png(fig)})

    # 2) distribución de scores
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    bins = np.linspace(0, 1, 26)
    if len(neg): ax.hist(neg, bins, color=GREY, alpha=0.75, label=f"no-palabra (n={len(neg)})")
    if len(pos): ax.hist(pos, bins, color=GREEN, alpha=0.75, label=f"tu palabra (n={len(pos)})")
    ax.axvline(thr, color=RED, ls="--", lw=1.5, label=f"umbral {thr:.2f}")
    ax.set_xlabel("score del modelo"); ax.set_ylabel("ejemplos de validación"); ax.legend(frameon=False)
    figs.append({"title": "Separación de scores",
                 "caption": "Lo ideal: gris pegado a 0, verde pegado a 1, con el umbral en medio sin tocar a ninguno.",
                 "img": _png(fig)})

    # 3) umbral vs errores
    sw = s["sweep"]
    if sw["frr"] and sw["fpr"]:
        fig, ax = plt.subplots(figsize=(6.4, 3.4))
        ax.plot(sw["thresholds"], np.array(sw["frr"]) * 100, color=RED, lw=2, label="palabras perdidas (FRR)")
        ax.plot(sw["thresholds"], np.array(sw["fpr"]) * 100, color=GREY, lw=2, label="falsas alarmas (FPR)")
        ax.axvline(thr, color="black", ls="--", lw=1, label=f"umbral elegido {thr:.2f}")
        ax.set_xlabel("umbral"); ax.set_ylabel("% de ejemplos"); ax.legend(frameon=False)
        figs.append({"title": "Qué pasa al mover el umbral",
                     "caption": "Subir el umbral reduce falsas alarmas pero pierde más palabras, y al revés. Es el equilibrio que ajustas.",
                     "img": _png(fig)})

    # 4) matriz de confusión
    c = s["counts"]
    m = np.array([[c["tp"], c["fn"]], [c["fp"], c["tn"]]])
    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    ax.imshow(m, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(m[i, j]), ha="center", va="center", fontsize=16,
                    color="white" if m[i, j] > m.max() / 2 else "black")
    ax.set_xticks([0, 1], ["detecta", "no detecta"]); ax.set_yticks([0, 1], ["tu palabra", "no-palabra"])
    ax.spines[:].set_visible(False)
    figs.append({"title": "Matriz de confusión (validación)",
                 "caption": "Arriba a la derecha: palabras perdidas. Abajo a la izquierda: falsas alarmas.",
                 "img": _png(fig)})

    # 5) composición de datos
    files, tr, va = d.get("files"), d.get("train_examples"), d.get("val_examples")
    if files and tr and va:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.4, 3.4))
        keys = ["wake_word", "negatives", "noise"]
        bars = a1.bar(keys, [files[k] for k in keys], color=[GREEN, GREY, BLUE])
        a1.bar_label(bars); a1.set_title("Archivos por carpeta")
        x = np.arange(2)
        b1 = a2.bar(x - 0.2, [tr["positive"], va["positive"]], 0.4, color=GREEN, label="positivos")
        b2 = a2.bar(x + 0.2, [tr["negative"], va["negative"]], 0.4, color=GREY, label="negativos")
        a2.bar_label(b1); a2.bar_label(b2)
        a2.set_xticks(x, ["entrenamiento", "validación"]); a2.set_title("Ejemplos (ventanas)")
        a2.legend(frameon=False)
        fig.tight_layout()
        figs.append({"title": "Composición de los datos",
                     "caption": "Los negativos y el ruido se trocean en ventanas, por eso hay más ejemplos que archivos.",
                     "img": _png(fig)})

    # 6) duración de las grabaciones
    durs = d.get("positive_durations_s")
    if durs:
        fig, ax = plt.subplots(figsize=(6.4, 3.4))
        ax.hist(durs, bins=min(15, max(5, len(durs) // 3)), color=GREEN, alpha=0.8)
        ax.axvline(s["window_seconds"], color=RED, ls="--", lw=1.5, label=f"ventana {s['window_seconds']:.2f} s")
        ax.set_xlabel("duración de la palabra (s)"); ax.set_ylabel("grabaciones"); ax.legend(frameon=False)
        figs.append({"title": "Duración de tus grabaciones",
                     "caption": "Deben ser parecidas entre sí. La ventana se calcula con ellas más un margen.",
                     "img": _png(fig)})
    return figs


# ───────────────────────────── HTML ─────────────────────────────

CSS = """
:root{--bg:#fff;--fg:#111827;--mut:#6b7280;--card:#f3f4f6;--line:#e5e7eb;--ok:#16a34a;--warn:#b45309;--info:#2563eb}
@media (prefers-color-scheme:dark){:root{--bg:#111827;--fg:#f3f4f6;--mut:#9ca3af;--card:#1f2937;--line:#374151;--warn:#fbbf24;--info:#60a5fa;--ok:#4ade80}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}
main{max-width:1000px;margin:0 auto;padding:24px 16px 64px}h1{margin:0 0 4px}h2{margin:36px 0 12px;font-size:19px}
.sub{color:var(--mut);margin:0 0 20px}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px}
.card{background:var(--card);border-radius:10px;padding:14px}.card .l{color:var(--mut);font-size:13px}
.card .v{font-size:26px;font-weight:600;margin:2px 0}.card .s{color:var(--mut);font-size:12px}
.advice{list-style:none;padding:0;margin:0}.advice li{padding:10px 12px;border-left:4px solid var(--info);background:var(--card);margin-bottom:8px;border-radius:0 8px 8px 0}
.advice .warn{border-color:var(--warn)}.advice .ok{border-color:var(--ok)}
.figs{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:20px}
figure{margin:0;background:var(--card);border-radius:10px;padding:12px}figure img{max-width:100%;background:#fff;border-radius:6px}
figcaption{color:var(--mut);font-size:13px;margin-top:6px}table{border-collapse:collapse;width:100%}
td,th{text-align:left;padding:6px 10px;border-bottom:1px solid var(--line)}th{color:var(--mut);font-weight:500}
details{background:var(--card);border-radius:10px;padding:10px 14px}pre{overflow:auto;font-size:12px}
.gloss dt{font-weight:600;margin-top:8px}.gloss dd{margin:0;color:var(--mut)}
"""


def _pct(x, nd=1):
    return "—" if x is None else f"{100 * x:.{nd}f}%"


def _table(rows, empty):
    if not rows:
        return f"<p class='sub'>{html.escape(empty)}</p>"
    body = "".join(f"<tr><td>{html.escape(r['file'])}</td><td>{r['score']:.3f}</td></tr>" for r in rows)
    return f"<table><tr><th>Archivo</th><th>Score</th></tr>{body}</table>"


def render_html(meta: dict, s: dict, figs: list[dict]) -> str:
    c, lat = s["counts"], s.get("latency")
    cards = [
        ("Palabras detectadas", _pct(s["recall"]),
         f"{c['tp']} de {c['val_positives']} · IC 95 %: no perdidas ≥ {_pct(1 - s['frr_ci95'][1], 0)}"),
        ("Palabras perdidas (FRR)", _pct(s["frr"]),
         f"IC 95 %: {_pct(s['frr_ci95'][0], 0)} – {_pct(s['frr_ci95'][1], 0)}"),
        ("Falsas alarmas (FPR)", _pct(s["fpr"], 2),
         f"IC 95 %: {_pct(s['fpr_ci95'][0], 1)} – {_pct(s['fpr_ci95'][1], 1)} · {c['fp']} de {c['val_negatives']}"),
        ("Falsas activaciones/hora",
         "—" if s["false_activations_per_hour"] is None else f"≈ {s['false_activations_per_hour']:.1f}",
         f"hasta {s['false_activations_per_hour_upper95']:.0f}/h (95 %) · sobre {s['val_negative_minutes']:.1f} min"
         if s["false_activations_per_hour_upper95"] is not None else ""),
        ("AUC", "—" if s["auc"] is None else f"{s['auc']:.3f}", "1.000 = separación perfecta"),
        ("Umbral", f"{s['threshold']:.2f}", f"ventana {s['window_seconds']:.2f} s"),
    ]
    if lat:
        cards.append(("Latencia (CPU, 1 hilo)", f"{lat['mean_ms']:.1f} ms",
                      f"p95 {lat['p95_ms']:.1f} ms · {lat['cpu_load_pct_at_hop']:.1f}% de un hilo"))
    size = []
    if s.get("onnx_size_kb"): size.append(f"{s['onnx_size_kb']:.0f} KB")
    if s.get("n_parameters"): size.append(f"{s['n_parameters'] / 1e3:.1f}K parámetros")
    if size:
        cards.append(("Tamaño del modelo", size[0], " · ".join(size[1:])))

    cards_html = "".join(f"<div class='card'><div class='l'>{html.escape(l)}</div>"
                         f"<div class='v'>{html.escape(v)}</div><div class='s'>{html.escape(sub)}</div></div>"
                         for l, v, sub in cards)
    advice = "".join(f"<li class='{a['level']}'>{html.escape(a['text'])}</li>" for a in s["advice"])
    figs_html = "".join(f"<figure><img alt='{html.escape(f['title'])}' src='data:image/png;base64,{f['img']}'>"
                        f"<figcaption><b>{html.escape(f['title'])}.</b> {html.escape(f['caption'])}</figcaption></figure>"
                        for f in figs)
    cfg_json = html.escape(json.dumps(meta.get("config", {}), indent=2))

    return f"""<!doctype html><html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>wakeforge · {html.escape(meta.get('name', ''))}</title><style>{CSS}</style></head><body><main>
<h1>Informe del modelo: {html.escape(meta.get('name', ''))}</h1>
<p class="sub">Generado el {s['generated_at']} · validación: {c['val_positives']} positivos y {c['val_negatives']} negativos</p>
<div class="cards">{cards_html}</div>
<h2>Avisos y recomendaciones</h2><ul class="advice">{advice}</ul>
<h2>Gráficas</h2><div class="figs">{figs_html}</div>
<h2>Palabras no detectadas</h2>{_table(s['missed_positives'], 'Ninguna en validación.')}
<h2>Falsas alarmas</h2>{_table(s['false_alarms'], 'Ninguna en validación.')}
<h2>Casos límite</h2>
<p class="sub">Las 5 palabras con score más bajo y los 5 no-palabra con score más alto: lo más cerca de fallar.</p>
{_table(s['weakest_positives'], '—')}<br>{_table(s['strongest_negatives'], '—')}
<h2>Cómo leerlo</h2><dl class="gloss">
<dt>FRR</dt><dd>De cada 100 veces que dices la palabra, cuántas no se detectan.</dd>
<dt>FPR</dt><dd>De cada 100 trozos de audio que no son la palabra, cuántos disparan el detector.</dd>
<dt>IC 95 %</dt><dd>Rango en el que probablemente está el valor real. Con pocos ejemplos es muy ancho.</dd>
<dt>Falsas activaciones/hora</dt><dd>Estimación aproximada; en uso real una activación puede disparar varias ventanas seguidas.</dd>
<dt>AUC</dt><dd>Qué tan bien se separan palabra y no-palabra sin depender del umbral.</dd></dl>
<h2>Configuración usada</h2><details><summary>Ver config</summary><pre>{cfg_json}</pre></details>
</main></body></html>"""


# ───────────────────────────── punto de entrada ─────────────────────────────


def generate_report(out_dir, *, meta: dict, losses, val_scores, val_labels, val_names,
                    onnx_path=None) -> Path:
    """Escribe stats/report.html, stats.json y raw.json. Devuelve la ruta del informe."""
    out_dir = Path(out_dir)
    stats_dir = out_dir / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)

    scores = np.asarray(val_scores, dtype=float)
    labels = np.asarray(val_labels).astype(int)
    losses = [float(x) for x in losses]
    names = list(val_names)

    (stats_dir / "raw.json").write_text(json.dumps(
        {"losses": losses, "val": {"names": names, "labels": labels.tolist(), "scores": scores.tolist()}}))

    onnx_path = Path(onnx_path) if onnx_path else out_dir / meta.get("model_file", "model.onnx")
    stats = compute_stats(meta, scores, labels, names, onnx_path)
    stats["advice"] = make_advice(meta, stats, losses)
    (stats_dir / "stats.json").write_text(json.dumps(stats, indent=2))

    try:
        figs = make_figures(meta, stats, losses, scores, labels)
    except ImportError:
        print("[stats] matplotlib no está instalado: solo se guardó stats.json (pip install matplotlib).")
        return stats_dir / "stats.json"

    report = stats_dir / "report.html"
    report.write_text(render_html(meta, stats, figs), encoding="utf-8")
    return report


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        sys.exit("Uso: python -m wakeforge.auto_statistics outputs/<nombre>")
    out = Path(argv[0])
    meta = json.loads((out / "metadata.json").read_text())
    raw = json.loads((out / "stats" / "raw.json").read_text())
    path = generate_report(out, meta=meta, losses=raw["losses"], val_scores=raw["val"]["scores"],
                           val_labels=raw["val"]["labels"], val_names=raw["val"]["names"])
    print(f"[stats] informe regenerado -> {path}")


if __name__ == "__main__":
    main()