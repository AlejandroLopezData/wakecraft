# wakeforge

Train your own custom wake word rapidly at home.

Record a few positive, negative, and noise samples.
Wakeforge comes with built-in noise samples and exports your trained model to ONNX, so you can start using your custom wake word in minutes.


## Why wakeforge?

- **Personal by design**: trained for your voice, your mic, your room.
- **Works with little data**: heavy augmentation + built-in noise.
- **Simple API**: train and detect in a handful of lines.
- **Portable**: exports to ONNX.

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