# record/ · Grabador de dataset para WakeForge

Herramienta de terminal para construir y mantener `my_recordings/`.
La lógica de audio reutilizable (formato, conversión, escritura de WAV,
estadísticas) vive en `src/wakeforge/audio.py`; aquí solo está la sesión
interactiva.

**Formato de salida:** WAV · PCM 16-bit · mono · 16 kHz.

## Instalación

Desde la raíz del proyecto:

```bash
pip install -e ".[record]"
```

Solo en Linux, `sounddevice` necesita PortAudio:

```bash
sudo apt install libportaudio2
```

## Uso

```bash
python -m record.recorder                                  # pregunta categoría y nº de tomas
python -m record.recorder --category wake_word --takes 50
python -m record.recorder -c negatives -n 100
python -m record.recorder -c noise -n 3                    # 3 grabaciones largas de ambiente
python -m record.recorder --list-devices                   # ver micrófonos disponibles
python -m record.recorder --input-device "USB"             # elegir micro por nombre o índice
```

Ejecútalo siempre desde la raíz del proyecto.

### Flujo de una sesión

```text
[12/50] wake_word   (archivos en carpeta: 11)
   ENTER = empezar   Q = salir
   [REC]    1.4s/3s [########--------] -18.2 dBFS  ENTER = terminar
   guardado wake_word_0012.wav  (1.4 s, pico -6.2 dBFS)
   [ENTER] siguiente  [R] repetir  [P] reproducir  [Q] salir
```

| Tecla | Acción |
|-------|--------|
| `ENTER` | Empezar una toma / terminarla / pasar a la siguiente |
| `R` | Borrar la última toma y repetirla con el mismo número |
| `P` | Reproducir la última toma (Ctrl+C corta la reproducción) |
| `Q` | Salir (las tomas ya guardadas se conservan) |

- El contador `[12/50]` cuenta las tomas de **esta sesión**; entre paréntesis
  se muestra cuántos archivos hay ya en la carpeta.
- `R` solo borra la toma que acaba de crear la sesión actual, nunca archivos
  anteriores.
- Si una toma es muy corta, casi silencio o está saturada, se avisa y se
  sugiere `R`. Nada se descarta automáticamente.

## Dónde se guarda

```text
my_recordings/
├── wake_word/   wake_word_0001.wav, wake_word_0002.wav, ...
├── negatives/   negatives_0001.wav, ...
└── noise/       noise_0001.wav, ...
```

- Las carpetas se crean solas si no existen.
- El siguiente nombre es siempre `máximo índice existente + 1`.
- Los archivos se crean en modo exclusivo: es imposible sobrescribir uno existente.

## Configuración (`record/config.toml`)

Dispositivos de audio, carpeta de salida y límites de duración por categoría
(`min_seconds` genera un aviso; `max_seconds` corta la captura). El formato de
audio no se configura aquí: está fijado en `wakeforge.audio`.

## Recomendaciones de grabado

- **wake_word**: una sola pronunciación por toma, con medio segundo de silencio
  antes y después. Varía distancia, tono y velocidad, y si puedes, varía de voz.
- **negatives**: frases normales, palabras parecidas a la wake word y habla
  cotidiana. Una frase por toma.
- **noise**: ambiente sin hablar (habitación, tele de fondo, ventilador, calle).
  Grabaciones largas; se trocean después en el pipeline de entrenamiento.

## Solución de problemas

| Síntoma | Qué probar |
|---------|-----------|
| `No se pudo importar 'sounddevice'` | `pip install -e ".[record]"` y, en Linux, `sudo apt install libportaudio2` |
| `No se pudo acceder al micrófono` | `--list-devices` y luego `--input-device <índice>` |
| Todo sale en silencio | Permiso de micrófono para el terminal (macOS: Ajustes > Privacidad; Windows: Privacidad > Micrófono) |
| WSL | El micrófono normalmente no está disponible; graba en el sistema anfitrión |
| Aviso de saturación | Baja la ganancia de entrada del micrófono en el sistema |