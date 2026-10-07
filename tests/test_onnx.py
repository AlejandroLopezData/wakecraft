import time
import numpy as np
import sounddevice as sd
import onnxruntime as ort


MODEL_PATH = "outputs/my_wakeword/model.onnx"

# Si entrenaste a 16 kHz:
SAMPLE_RATE = 16000

# Pon aquí la duración de la ventana que usaste al entrenar.
WINDOW_SECONDS = 1.0

# Cada cuánto analizamos el micro
STEP_SECONDS = 0.25

# Ajusta esto después de ver los scores
THRESHOLD = 0.5


# ============================================================
# ONNX
# ============================================================

print("Cargando modelo...")

session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"],
)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

print(f"Modelo: {MODEL_PATH}")
print(f"Input : {input_name}")
print(f"Output: {output_name}")


# ============================================================
# AUDIO
# ============================================================

WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_SECONDS)
STEP_SAMPLES = int(SAMPLE_RATE * STEP_SECONDS)

buffer = np.zeros(WINDOW_SAMPLES, dtype=np.float32)


def predict(audio):
    # [samples] -> [1, samples]
    x = audio.reshape(1, -1).astype(np.float32)

    output = session.run(
        [output_name],
        {
            input_name: x
        },
    )[0]

    return float(np.asarray(output).reshape(-1)[0])


# ============================================================
# MICRO
# ============================================================

print()
print("🎤 Abriendo micrófono...")
print(f"Sample rate: {SAMPLE_RATE}")
print(f"Ventana: {WINDOW_SECONDS}s")
print()
print("Habla la wake word...")
print("Ctrl+C para salir.")
print()

last_detection = 0.0

try:
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=STEP_SAMPLES,
    ) as stream:

        while True:

            # Leer audio del micro
            data, overflowed = stream.read(STEP_SAMPLES)

            # [samples, 1] -> [samples]
            chunk = data[:, 0]

            # Desplazar buffer
            n = len(chunk)

            buffer[:-n] = buffer[n:]
            buffer[-n:] = chunk

            # Ejecutar ONNX
            score = predict(buffer)

            # Mostrar score
            print(
                f"\rScore: {score:.4f}",
                end="",
                flush=True,
            )

            # Detectar wake word
            now = time.time()

            if score >= THRESHOLD and now - last_detection > 1.5:

                print()
                print()
                print("🔥🔥🔥 WAKE WORD DETECTADA 🔥🔥🔥")
                print()

                last_detection = now


except KeyboardInterrupt:
    print()
    print()
    print("Micro cerrado.")

except Exception as e:
    print()
    print()
    print("ERROR:")
    print(e)