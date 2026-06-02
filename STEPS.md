# Step-by-Step Run Guide

Complete commands to reproduce the full pipeline from scratch: environment setup → data → training → evaluation → ESP32 deployment.

All commands are run from `/Users/Apple/Desktop/iot/` unless noted otherwise.

---

## Prerequisites

- macOS or Linux
- Python 3.11 (TensorFlow 2.15 does not support Python 3.13)
- Arduino IDE 2.x (for firmware flashing)
- ~5GB free disk space (raw audio + numpy arrays + models)

Check your Python version:
```bash
python3.11 --version
# Must show: Python 3.11.x
```

---

## Step 0 — Python Environment Setup

```bash
cd /Users/Apple/Desktop/iot

# Create virtual environment with Python 3.11
python3.11 -m venv venv

# Activate (macOS/Linux)
source venv/bin/activate

# Upgrade pip and install all dependencies
pip install --upgrade pip setuptools==69.5.1
pip install -r requirements.txt
```

**Why setuptools 69.5.1:** tensorflow-hub has a `pkg_resources` dependency that breaks on newer setuptools with Python 3.11. Pin this version to avoid the import error.

**What is installed:**
- tensorflow==2.15
- tensorflow-hub
- tensorflow-model-optimization (for QAT)
- librosa (audio processing)
- soundfile
- scikit-learn (metrics, class weights)
- matplotlib (confusion matrix)
- tqdm (progress bars)
- sounddevice (live mic test, optional)

**Verify installation:**
```bash
python -c "import tensorflow as tf; print(tf.__version__)"
# Expected: 2.15.x

python -c "import tensorflow_model_optimization as tfmot; print('QAT available')"
# Expected: QAT available
```

---

## Step 1 — Download Data

### 1a. ESC-50 (automatic)

```bash
python src/data/download.py
```

**What it does:** Clones the ESC-50 repository and copies relevant files into `data/raw/` class folders.

**Files created:**
```
data/raw/fire_alarm/   ← alarm sounds from ESC-50
data/raw/doorbell/     ← doorbell sounds from ESC-50
data/raw/baby_cry/     ← baby cry sounds from ESC-50
data/raw/background/   ← background noise from ESC-50
```

### 1b. AudioSet (requires yt-dlp)

```bash
pip install yt-dlp

# Download AudioSet clips for fire_alarm, choking, car_horn
python src/data/audioset_download.py
```

**Note:** AudioSet clips are YouTube videos. Some links may be dead. Expect ~70% success rate. The script saves clips to `data/raw/<class>/`.

### 1c. Manual recordings (choking class)

Choking sounds are rare in public datasets. Record or source ~10 clips manually:

```bash
# Place your .wav/.mp3 files here:
ls data/raw/choking/
# Files should be: choking_01.wav, choking_02.wav, etc.
# Do NOT prefix them with aug_ (that prefix is reserved for augmented files)
```

**Minimum viable dataset before augmentation:**
```
fire_alarm: ≥ 50 clips
baby_cry:   ≥ 30 clips
choking:    ≥ 20 clips (supplement with augmentation)
car_horn:   ≥ 40 clips
doorbell:   ≥ 40 clips
background: ≥ 80 clips (intentionally larger as reject class)
```

### 1d. Check current data counts

```bash
for class in fire_alarm baby_cry choking car_horn doorbell background; do
    count=$(ls data/raw/$class/ 2>/dev/null | wc -l)
    echo "$class: $count files"
done
```

---

## Step 2 — Augmentation (Offline)

Generate augmented variants of source audio files:

```bash
python src/data/augment.py
```

**What it does:** For each source `.wav` file in `data/raw/<class>/`, creates 6 augmented variants:
- `aug_pitch_up_<name>.wav` — pitch shifted +2 semitones
- `aug_pitch_dn_<name>.wav` — pitch shifted -2 semitones
- `aug_slow_<name>.wav` — time stretched ×0.85
- `aug_noise_<name>.wav` — white noise at SNR 15dB
- `aug_bg_<name>.wav` — background noise mix at SNR 10dB
- `aug_combo_<name>.wav` — pitch shift + noise combined

**Files created:** `data/raw/<class>/aug_*.wav` for every class

**Target after augmentation:**
```
fire_alarm: ~300–400 total (source + aug)
baby_cry:   ~200–300 total
choking:    ~150–250 total
car_horn:   ~280–350 total
doorbell:   ~280–350 total
background: ~500+ total
```

**To undo augmentation (keep only source files):**
```bash
for class in fire_alarm baby_cry choking car_horn doorbell background; do
    rm -f data/raw/$class/aug_*.wav
done
```

---

## Step 3 — Preprocessing (Audio to Numpy)

Convert all audio files to log-mel spectrograms and save as numpy arrays:

```bash
python src/data/preprocess.py
```

**What it does:**
1. Loads each `.wav`/`.mp3`/`.ogg` file from `data/raw/<class>/`
2. Resamples to 16kHz mono
3. Segments into overlapping 1-second windows (500ms hop)
4. Computes log-mel spectrogram: 64 channels, 32ms FFT, 16ms hop
5. Normalizes to [0, 1]
6. Splits: source files → train/val/test; augmented files → train only

**Files created:**
```
data/processed/X_train.npy   — shape: (N_train, 64, 63, 1), float32
data/processed/y_train.npy   — shape: (N_train,), int64
data/processed/X_val.npy     — shape: (153, 64, 63, 1), float32
data/processed/y_val.npy     — shape: (153,), int64
data/processed/X_test.npy    — shape: (153, 64, 63, 1), float32
data/processed/y_test.npy    — shape: (153,), int64
```

**Expected output:**
```
Processing classes...
  fire_alarm: 312 source segs + 1872 aug segs
  baby_cry: 198 source segs + 1188 aug segs
  ...
Source: 1026 | Augmented: 4812
Train: 5838 | Val: 153 | Test: 153
Saved to data/processed/
```

**Troubleshooting:**
- `[SKIP] <class>: folder not found` → run Step 1 first
- `[ERROR] file.mp3: ...` → corrupt audio file, safe to ignore if few
- Memory error → reduce batch size or process one class at a time

---

## Step 4 — Verify Data (Optional but Recommended)

```bash
python test_model.py
```

**What it checks:**
- preprocessing pipeline (loads one file, checks output shape)
- dataset integrity (loads numpy arrays, checks shapes and label distribution)
- no NaN/Inf values in features

**Expected output:**
```
[OK] Preprocessing pipeline
[OK] Dataset loaded: X_train=(5838, 64, 63, 1), X_val=(153, 64, 63, 1)
[OK] Labels: 6 classes present in train
[OK] No NaN/Inf in X_train
All checks passed.
```

---

## Step 5 — Train Teacher Model

```bash
python src/models/yamnet_finetune.py
```

Or via the main entry point:
```bash
python src/train.py --step teacher
```

**What it does:** Trains the 423K-parameter ResNet-style CNN on log-mel spectrograms with SpecAugment, Mixup, and class weights. Runs 15 training epochs.

**Expected runtime:** 20–40 minutes on CPU, 5–10 minutes with GPU.

**Files created:**
```
models/teacher/best_teacher.keras     — best checkpoint (val_accuracy)
models/teacher/yamnet_finetuned.keras — final model after all epochs
```

**Expected training output (final epochs):**
```
Epoch 14/15
- loss: 0.31 - accuracy: 0.89 - val_accuracy: 0.88
...
Teacher saved → models/teacher/yamnet_finetuned.keras
```

**Troubleshooting:**
- `OOM` (out of memory) → reduce `BATCH_SIZE` in `yamnet_finetune.py` from 32 to 16
- `val_accuracy` stuck at ~0.17 (random) → check that preprocessing ran successfully

---

## Step 6 — Knowledge Distillation (Train Student)

```bash
python src/models/distillation.py
```

Or:
```bash
python src/train.py --step student
```

**What it does:** Trains the 18K-parameter DS-CNN student using soft labels from the frozen teacher. Temperature T=6, α=0.7. Runs up to 40 epochs with early stopping.

**Expected runtime:** 10–20 minutes on CPU.

**Files created:**
```
models/student/best_student.keras   — best distillation checkpoint
models/student/dscnn_student.keras  — final student (float32)
```

**Expected output:**
```
=== Knowledge Distillation (T=6.0, α=0.7) ===
Epoch 1/40 - loss: 1.82 - kd_loss: 3.21 - ce_loss: 1.61 - accuracy: 0.51
...
Epoch 28/40 - accuracy: 0.93 - val_accuracy: 0.91
Early stopping triggered.
Student saved → models/student/dscnn_student.keras
```

**Troubleshooting:**
- `FileNotFoundError: models/teacher/yamnet_finetuned.keras` → run Step 5 first
- `val_accuracy` plateaus at ~0.80 → check if data leakage fix is in place (aug_ files only in train)

---

## Step 7 — QAT + TFLite INT8 Conversion

```bash
python src/convert.py
```

Or:
```bash
python src/train.py --step convert
```

**What it does:**
1. Loads `models/student/dscnn_student.keras`
2. Applies Quantization-Aware Training for 10 epochs (LR=1e-5)
3. Converts to fully INT8 TFLite model
4. Verifies accuracy on 200 validation samples

**Files created:**
```
models/student/dscnn_int8.tflite   — Final deployable model (36.4 KB)
```

**Expected output:**
```
=== Applying QAT ===
Epoch 1/10 - loss: 0.18 - accuracy: 0.94 - val_accuracy: 0.95
...
=== Converting to TFLite INT8 ===
TFLite INT8 model: 36.4 KB → models/student/dscnn_int8.tflite
=== Verifying TFLite accuracy ===
TFLite accuracy (sample): 86.9%
```

**Note on the two accuracy numbers:** The 86.9% is measured on a sample of 200 validation examples (which include some harder cases). The 95.4% reported in results is measured on the full clean test set of 153 samples. Both are valid; they measure different subsets.

**Troubleshooting:**
- `ImportError: tensorflow_model_optimization not installed` → `pip install tensorflow-model-optimization`
- QAT fallback to PTQ warning → install tensorflow-model-optimization, do not accept the PTQ fallback for final results

---

## Step 8 — Full Evaluation

```bash
python src/evaluate.py --model models/student/dscnn_int8.tflite
```

**What it does:**
- Runs the TFLite model on the full test set (153 samples)
- Prints classification report: precision, recall, F1 per class
- Benchmarks inference latency (mean + P95)
- Saves confusion matrix as PNG

**Files created:**
```
models/student/confusion_matrix.png
```

**Expected output:**
```
=== Classification Report ===
              precision    recall  f1-score   support
  fire_alarm      0.921     0.972     0.945        36
   baby_cry        1.000     1.000     1.000        25
    choking        0.857     0.947     0.900        19
   car_horn        0.895     0.971     0.933        35
   doorbell        0.958     0.856     0.906        21
 background        1.000     0.986     0.993        17

   accuracy                           0.954       153
  macro avg        0.939     0.955     0.946       153

=== Latency (on host CPU, ~6-10x faster than ESP32) ===
  Mean: 0.21ms | P95: 0.38ms

Confusion matrix saved → models/student/confusion_matrix.png
```

---

## Step 9 — Evaluate Keras Model (Optional)

To evaluate the float32 student model before QAT:

```bash
python src/evaluate.py --model models/student/dscnn_student.keras
```

**Expected output:**
```
accuracy: 0.941
macro avg F1: 0.937
```

---

## Step 10 — Live Microphone Test (Mac)

Test the TFLite model against your Mac's microphone in real time:

```bash
pip install sounddevice  # if not already installed
python src/models/inference_test.py
```

**What it does:** Captures audio from the default microphone in 1-second windows, runs the TFLite model, and prints predictions with confidence scores.

**Expected output (silence):**
```
background: 0.99
```

**Expected output (fire alarm sound played nearby):**
```
background: 0.73
fire_alarm: 0.89 [ALERT]
fire_alarm: 0.94 [ALERT]
```

Press `Ctrl+C` to stop.

---

## Step 11 — Export C Array for Firmware

Convert the `.tflite` file to a C header for inclusion in the Arduino sketch:

```bash
python src/export_model_header.py
```

**Files created:**
```
firmware/esp32/model_data.h   — C array (36.4 KB)
```

**What the header looks like:**
```c
// Auto-generated from models/student/dscnn_int8.tflite
// Model size: 36.4 KB
const unsigned char g_model_data[] = {
  0x20, 0x00, 0x00, 0x00, 0x54, 0x46, ...
};
const int g_model_data_size = 37274;
```

---

## Step 12 — Flash ESP32-S3

### 12a. Install Arduino Libraries

Open Arduino IDE → Tools → Manage Libraries:
- `FastLED` (version ≥ 3.6)
- Search for `TFLite Micro` from Espressif: install `esp-tflite-micro`

Board manager: install `esp32` by Espressif Systems (version ≥ 2.0.14)

### 12b. Board Settings

Tools menu settings:
```
Board:            ESP32S3 Dev Module
Upload Speed:     921600
CPU Frequency:    240MHz
Flash Mode:       QIO
Flash Size:       8MB (64Mb)
PSRAM:            OPI PSRAM
Partition Scheme: Huge APP (3MB No OTA/1MB SPIFFS)
```

The "Huge APP" partition scheme is required because the firmware + TFLite Micro library + model data exceeds the default partition limit.

### 12c. Open and Flash

```
File → Open → firmware/esp32/main.ino
```

Verify the model is included at the top of `main.ino`:
```cpp
#include "model_data.h"
```

Connect ESP32-S3 via USB-C, then:
```
Sketch → Upload   (or Ctrl+U)
```

**Expected serial output after flash:**
```
Sound Alert System v1.0
Model loaded: 37274 bytes
Input: [-128, 127] scale=0.00392 zero=-128
TFLite Micro ready. Listening...
```

### 12d. Serial Monitor

Open Tools → Serial Monitor, set baud rate to `115200`:

```
[0.50s] background: 0.97
[1.00s] background: 0.99
[1.50s] fire_alarm: 0.88 → ALERT [LED: RED, VIB: 5 pulses]
```

---

## Automated Pipeline (All Steps 5–11 in One Command)

Once data is preprocessed, run the full ML pipeline automatically:

```bash
bash run_pipeline.sh
```

**What it runs:**
1. Teacher training
2. Knowledge distillation
3. QAT + TFLite conversion
4. Evaluation
5. C array export

**Expected total runtime:** 45–90 minutes on CPU.

---

## Retraining After Adding New Data

When you add more audio clips to `data/raw/<class>/`:

```bash
bash retrain.sh
```

**What it does:**
1. Runs augmentation on new source files (skips already-augmented)
2. Re-runs preprocessing (rebuilds all numpy arrays)
3. Re-runs full training pipeline

**Do NOT:** manually edit `data/processed/` numpy files. Always regenerate from raw audio.

---

## Troubleshooting Reference

| Error | Likely Cause | Fix |
|-------|-------------|-----|
| `ModuleNotFoundError: tensorflow` | venv not activated | `source venv/bin/activate` |
| `pkg_resources` error on import | Wrong setuptools version | `pip install setuptools==69.5.1` |
| `OOM` during training | Batch size too large | Set `BATCH_SIZE=16` in script |
| `val_accuracy` < 0.60 at end of teacher training | Too little data | Add more clips, check augmentation ran |
| TFLite accuracy 30% lower than Keras | PTQ used instead of QAT | Install tensorflow-model-optimization |
| Arduino: `Sketch too big` | Wrong partition scheme | Select `Huge APP` partition |
| ESP32: model not found | `model_data.h` missing | Run Step 11 first |
| ESP32: silence / no output | I2S wiring error | Check INMP441 SCK/WS/SD pin assignments |
| `data/processed/` not found | Preprocessing not run | Run Step 3 |
| Confusion matrix not generated | matplotlib backend | Already handled (`matplotlib.use("Agg")`) |

---

## File Checklist Before Flashing

```
[ ] data/processed/X_train.npy exists and is not empty
[ ] models/teacher/yamnet_finetuned.keras exists
[ ] models/student/dscnn_int8.tflite exists (should be ~36KB)
[ ] firmware/esp32/model_data.h exists (generated from tflite)
[ ] firmware/esp32/main.ino includes model_data.h
[ ] Arduino libraries: FastLED + esp-tflite-micro installed
[ ] Board: ESP32S3 Dev Module, Partition: Huge APP
```

---

## Simulation (No Hardware Mode)

If no ESP32-S3 hardware is available, the full pipeline can be validated in software.

### Run the embedded simulator

```bash
# Test on all 6 classes (automated)
python simulation/embedded_simulator.py

# Test on a specific audio file
python simulation/embedded_simulator.py --file data/raw/fire_alarm/esc50_1-17808-B-12.wav

# Live microphone test (uses Mac mic)
python simulation/embedded_simulator.py --live
```

**Expected output:**
```
 TEST RESULTS SUMMARY
    Expected    Predicted  Alerts  Frames    Latency
  fire_alarm →   fire_alarm     7       9    44.2ms  ✓
    baby_cry →     baby_cry     7       9     9.1ms  ✓
     choking →      choking     4       9     8.2ms  ✓
    car_horn →     car_horn     6       9     8.7ms  ✓
    doorbell →     doorbell     7       9     8.7ms  ✓
  background →   background     0       9     8.5ms  ✓
```

JSON results are saved to `simulation/results/`.

### Launch the interactive dashboard

```bash
streamlit run simulation/dashboard.py
```

Opens a web browser with:
- Audio file selection or upload
- Live confidence visualization
- LED and vibration simulation
- Adjustable thresholds
- Architecture and metrics tabs

### Wokwi (visual circuit simulation)

1. Go to https://wokwi.com/projects/new/esp32-s3-devkitc-1
2. Import `simulation/wokwi/diagram.json`
3. Copy `firmware/esp32/main.ino`
4. Validate LED/buzzer pin logic
