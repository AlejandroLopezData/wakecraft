<div align="center">

# WakeTorch

**Wake word training; train a personal wake word model in minutes, in any language.**

Guided CLI · Real voice samples · Fast training · CPU inference · ONNX export  
Open source · Local · Lightweight

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) ![ONNX](https://img.shields.io/badge/ONNX-005CED?style=flat&logo=onnx&logoColor=white) ![macOS](https://img.shields.io/badge/macOS-000000?style=flat&logo=apple&logoColor=white) ![Windows](https://img.shields.io/badge/Windows-0078D4?style=flat&logo=windows&logoColor=white) ![Linux](https://img.shields.io/badge/Linux-FCC624?style=flat&logo=linux&logoColor=black) ![Raspberry Pi](https://img.shields.io/badge/Raspberry%20Pi-C51A4A?style=flat&logo=raspberrypi&logoColor=white) ![License](https://img.shields.io/badge/License-MIT-green?style=flat) ![Status](https://img.shields.io/badge/Status-Early%20Alpha-orange?style=flat)

**Any language. Your voice. Your wake word.**

[Model & Documentation](https://alejandrolopezdata.github.io/waketorch/)

</div>

**Any language. Your voice. Your wake word.**

[📚 Model & Documentation](https://alejandrolopezdata.github.io/waketorch/)

</div>

## ✨ Why Wakeforge?

* **🎙️ Personal by design** — trained for your voice, microphone, and environment.
* **📊 Works with little data** — heavy augmentation and built-in noise make small datasets useful.
* **⚡ Simple API** — train and detect a wake word with just a few lines of code.
* **📦 Portable** — export your trained model to ONNX and use it wherever you need.

## 🔧 How it works

Wakeforge uses a simple pipeline to turn a few recordings into a practical wake-word detector.

### 1. Positive samples

Your recordings are combined with TTS-generated voices and augmented using:

* Noise
* Reverb
* Speed changes
* Pitch changes
* Gain variations

### 2. Negative samples

The training pipeline combines:

* Built-in negative samples
* Your own recordings
* Automatically generated **hard negatives**

### 3. Noise

Built-in environmental noise and your own recordings are mixed into the dataset to make the model more robust to real-world conditions.

### 4. Model

Wakeforge uses a **small CNN operating on mel-spectrograms**, designed to train on a CPU in just a few minutes.

### 5. Detection

The trained model runs using a **streaming sliding window**, with smoothing and a calibrated detection threshold to reduce false positives.

> Real recordings are weighted more heavily than synthetic samples.
>
> **For the best results, record at least 10–20 samples of your own voice.**

## 🎯 Improving quality

The more realistic and diverse your dataset is, the better the detector can perform.

Consider adding:

* Different voices from friends or family
* Multiple TTS voices
* More negative speech samples
* TV and music
* Similar-sounding words
* Background noise from your actual environment

The goal is to make the training data resemble the situations where you will actually use the wake word.

## 🗺️ Roadmap

* [ ] Guided recording tool
* [ ] Data pipeline and augmentation
* [ ] Small model and training loop
* [ ] Streaming detector
* [ ] TTS backends — Piper first
* [ ] Hard negative generation
* [ ] Negative/noise dataset download scripts
* [ ] Threshold calibration
* [ ] False-positives-per-hour evaluation
* [ ] ONNX export
* [ ] Documentation

## 📄 License

**Code:** Apache-2.0

Datasets and TTS voices may have their own licenses. Check the applicable licenses before redistributing models or using them commercially.

License details will be available in:

`docs/licenses.md`

*(Coming soon.)*
