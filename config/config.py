# ──────────────────────────────────────────────────────────────
# wakeforge · configuración de entrenamiento
# Todo es opcional: si borras una línea se usa su valor por defecto.
# También puedes cambiar valores sin editar el archivo:
#   python scripts/train.py --set training.steps=4000 --set model.width=32
# ──────────────────────────────────────────────────────────────

[project]
name = "my_wakeword"            # nombre de la carpeta de salida: outputs/<name>/
recordings_dir = "my_recordings"
output_dir = "outputs"

[audio]
sample_rate = 16000
window_seconds = 0.0            # 0 = automática (según la duración de tus grabaciones)
window_margin = 0.4             # segundos extra sobre la duración típica de la palabra
window_min = 1.0                # límites de la ventana automática
window_max = 3.0
window_percentile = 90          # percentil de duración usado para calcular la ventana
trim_silence = true             # recorta silencios al inicio y final de tus grabaciones
trim_top_db = 30.0              # umbral de silencio (dB por debajo del pico)

[features]                      # espectrograma mel (va DENTRO del ONNX)
n_mels = 40
n_fft = 512
win_length = 400
hop_length = 160
f_min = 20.0
f_max = 7600.0

[model]
width = 64                      # canales: 32 = más ligero, 96 = más capacidad
kernel_size = 9
dilations = [1, 2, 4, 1, 2, 4]  # una entrada por bloque; más bloques = más profundo
dropout = 0.1

[training]
steps = 2000                    # pasos de entrenamiento (con pocos datos, 1000-3000 basta)
batch_size = 64
lr = 0.002
weight_decay = 0.01
val_fraction = 0.2              # parte de tus archivos reservada para medir (por archivo)
positive_ratio = 0.33           # fracción de positivos en cada batch
label_smoothing = 0.01
ema_decay = 0.99                # media móvil de pesos (0.999 si entrenas muchos pasos)
eval_every = 100                # cada cuántos pasos se evalúa
max_fpr = 0.005                 # falsos positivos admitidos (por ventana) al elegir el umbral
min_threshold = 0.5             # el umbral recomendado nunca baja de este valor
grad_clip = 5.0
negative_stride = 0.5           # solape al trocear negativos largos (0.5 = mitad de ventana)
seed = 0
num_workers = 0                 # 0 funciona en todos los sistemas; sube si tienes CPU de sobra
device = "auto"                 # "auto" | "cpu" | "cuda"

[augmentation]
gain_db = 6.0                   # ganancia aleatoria ±dB
noise_prob = 0.8                # probabilidad de mezclar ruido de fondo
snr_min_db = 0.0                # SNR mínimo (más bajo = más ruido, más difícil)
snr_max_db = 20.0
speed_factors = [0.9, 0.95, 1.0, 1.05, 1.1]
speed_prob = 0.5
freq_mask = 6                   # SpecAugment (0 = desactivado)
time_mask = 15

[export]
filename = "model.onnx"
opset = 17
verify = true                   # compara PyTorch vs ONNX Runtime tras exportar