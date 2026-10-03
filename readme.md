# wakeforge

Train your own custom wake word at home, no ML knowledge required.

Record a few samples (or none), pick your language, and let multilingual TTS
generate positives and *hard negatives*. wakeforge ships with negative and
noise banks, and exports to ONNX so you can use your wake word in a few lines
of code.

> ⚠️ Early development: the API will change.

## Why wakeforge?

- **Personal by design**: trained for your voice, your mic, your room.
- **Works with little data**: heavy augmentation + built-in negatives and noise.
- **Multilingual**: TTS-generated positives and hard negatives in your language.
- **Simple API**: train and detect in a handful of lines.
- **Portable**: exports to ONNX.

## Quickstart (planned API)

```python
from wakeforge import train, Detector

train(
    wake_word="hey computer",
    language="en",
    recordings="my_recordings/",       # optional
    extra_negatives="my_negatives/",   # optional
    extra_noise="my_noise/",           # optional
)

Detector("hey_computer").listen(on_detect=lambda: print("Activated!"))
```

## How it works

1. **Positives**: your recordings + TTS-generated voices, with augmentation
   (noise, reverb, speed, pitch, gain).
2. **Negatives**: built-in bank + your own + automatically generated hard negatives.
3. **Noise**: built-in bank + your own, mixed into everything.
4. **Model**: a small CNN over mel-spectrograms, trained on CPU in minutes.
5. **Detection**: streaming sliding window with smoothing and a calibrated threshold.

Real recordings are weighted higher than synthetic ones. For best results,
record at least 10-20 samples yourself.

## Improving quality

Add more voices (friends, family, TTS voices), more negatives (normal speech,
TV, music, similar-sounding words) and noise from your own environment.

## Roadmap

- [ ] Guided recording tool
- [ ] Data pipeline and augmentation
- [ ] Small model and training loop
- [ ] Streaming detector
- [ ] TTS backends (Piper first)
- [ ] Hard negative generation
- [ ] Negative/noise dataset download scripts
- [ ] Threshold calibration and false-positives-per-hour evaluation
- [ ] ONNX export and docs

## Licenses

Code: Apache-2.0. Datasets and TTS voices have their own licenses, so check
them before redistributing models or using them commercially.
Details in `docs/licenses.md` (coming soon).