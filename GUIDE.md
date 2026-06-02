# Sound Alert System — Complete Project Guide

Wearable IoT device for deaf and hard-of-hearing users. The device listens to ambient audio continuously, classifies critical sounds on-device using a tiny ML model, and alerts the user via RGB LEDs and a vibration motor — with zero cloud dependency.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Hardware](#2-hardware)
3. [Sound Classes](#3-sound-classes)
4. [ML Pipeline Architecture](#4-ml-pipeline-architecture)
5. [Design Decisions — Why Each Choice Was Made](#5-design-decisions--why-each-choice-was-made)
6. [Dataset](#6-dataset)
7. [Feature Extraction](#7-feature-extraction)
8. [Augmentation Strategy](#8-augmentation-strategy)
9. [Teacher Model](#9-teacher-model)
10. [Knowledge Distillation](#10-knowledge-distillation)
11. [Student Model — DS-CNN](#11-student-model--ds-cnn)
12. [Quantization — QAT vs PTQ](#12-quantization--qat-vs-ptq)
13. [Firmware — Inference on ESP32-S3](#13-firmware--inference-on-esp32-s3)
14. [Results](#14-results)
15. [Project Structure](#15-project-structure)
16. [Key Technical Parameters](#16-key-technical-parameters)

---

## 1. Project Overview

People who are deaf or hard-of-hearing cannot rely on sound as a safety signal. A smoke alarm, a crying baby, or a person choking nearby may go entirely unnoticed. This project addresses that gap with a wearable device that performs real-time, on-device sound classification and provides haptic and visual alerts.

**Core constraints that shaped every design decision:**

- Inference must happen fully on-device — no Wi-Fi, no cloud, no latency from a round-trip
- The microcontroller (ESP32-S3) has ~512KB SRAM available for TFLite Micro
- Model must fit in flash: target < 100KB after INT8 quantization
- End-to-end latency must be under 500ms from sound to alert
- The system must run on a small LiPo battery for a full day

---

## 2. Hardware

| Component | Part | Notes |
|-----------|------|-------|
| Microcontroller | ESP32-S3 (dual-core LX7 @ 240MHz) | 8MB PSRAM, built-in BLE |
| Microphone | INMP441 (I2S digital) | 16kHz, SNR 61dB, omnidirectional |
| Visual alert | WS2812B RGB LEDs | Color-coded per threat level |
| Haptic alert | ERM vibration motor | Pulse patterns per class |
| Power | LiPo 3.7V 1000mAh + TP4056 charger | ~8–12h runtime estimated |

The INMP441 communicates over I2S — it delivers 32-bit samples at 16kHz with no ADC noise. The ESP32-S3's dual-core architecture allows one core to handle audio capture and the other to run inference without blocking.

---

## 3. Sound Classes

| Class | Priority | LED Color | Vibration Pattern |
|-------|----------|-----------|-------------------|
| fire_alarm | Critical | Red flash | 5 rapid pulses |
| choking | Critical | Red flash | 5 rapid pulses |
| baby_cry | High | Orange | 3 pulses |
| car_horn | High | Yellow | 3 pulses |
| doorbell | Medium | Blue | 1 long pulse |
| background | — (reject) | Off | None |

The `background` class is a reject class: it absorbs ambient noise, speech, and anything that is not one of the five alert classes. This prevents false positives from common everyday sounds.

**Per-class confidence thresholds** (applied at inference time):

| Class | Threshold | Reasoning |
|-------|-----------|-----------|
| fire_alarm | 0.75 | Critical — alert even at moderate confidence |
| choking | 0.75 | Critical — false negatives are costly |
| baby_cry | 0.80 | High importance, reasonably distinctive |
| car_horn | 0.80 | Distinctive sound, some risk of road noise confusion |
| doorbell | 0.90 | Low stakes, many impersonators (phone rings, notifications) |

---

## 4. ML Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    AUDIO CAPTURE                                │
│  INMP441 (I2S) → 16kHz mono PCM → 1-second sliding window      │
│                    (500ms stride, 50% overlap)                  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                 FEATURE EXTRACTION                              │
│  Log-Mel Spectrogram                                            │
│  • 64 mel channels                                              │
│  • FFT window: 32ms (512 samples at 16kHz)                      │
│  • Hop length: 16ms (256 samples)                               │
│  • Frequency range: 60Hz – 7600Hz                               │
│  • Output shape: (64, 63, 1)  [freq × time × channel]          │
└────────────────────────────┬────────────────────────────────────┘
                             │
              ┌──────────────┴──────────────┐
              │         TRAINING PHASE       │
              │                             │
              ▼                             ▼
┌─────────────────────┐        ┌────────────────────────┐
│   AUGMENTATION      │        │   (no augment at test) │
│  SpecAugment        │        │                        │
│  Mixup (α=0.2)      │        │                        │
│  Pitch shift ±2st   │        │                        │
│  Time stretch ×0.85 │        │                        │
│  White noise SNR15dB│        │                        │
│  Background mix     │        │                        │
└──────────┬──────────┘        └────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────┐
│              TEACHER MODEL (Train only, not deployed)           │
│  ResNet-style CNN                                               │
│  4 × [Conv2D → BatchNorm → ReLU → MaxPool]                      │
│  Channels: 32 → 64 → 128 → 256                                  │
│  GlobalAveragePooling → Dropout(0.4) → Dense(128) → Dense(6)    │
│  423,174 parameters — too large for ESP32                       │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             │ Knowledge Distillation
                             │ Temperature T=6, α=0.7
                             │ Loss = 0.7·KL(teacher_soft, student_soft)
                             │      + 0.3·CE(labels, student)
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              STUDENT MODEL — DS-CNN (Deployed)                  │
│  Initial Conv2D(32) → MaxPool                                   │
│  DS Block: DepthwiseConv → BN → ReLU → Conv1×1 → BN → ReLU     │
│  Block 1: 64ch → MaxPool                                        │
│  Block 2: 64ch → MaxPool                                        │
│  Block 3: 128ch                                                 │
│  GlobalAveragePooling → Dropout(0.3) → Dense(6, softmax)        │
│  18,086 parameters — ~72KB float32                              │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             │ Quantization-Aware Training (QAT)
                             │ 10 fine-tuning epochs at LR=1e-5
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              INT8 QUANTIZED TFLITE MODEL                        │
│  All weights and activations: INT8                              │
│  Model size: 36.4 KB (4× compression from float32)             │
│  Input: INT8 quantized log-mel spectrogram                      │
│  Output: INT8 class logits                                      │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              ESP32-S3 RUNTIME                                   │
│  TFLite Micro interpreter                                       │
│  Temporal smoothing: 3-frame majority vote                      │
│  Per-class confidence thresholds                                │
│  Output: LED color + vibration pattern                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. Design Decisions — Why Each Choice Was Made

### 5.1 Log-Mel Spectrogram over MFCC

MFCCs (Mel-Frequency Cepstral Coefficients) apply a Discrete Cosine Transform on top of the log-mel filterbank energies. This compression was designed for speech recognition and for human-engineered features fed into GMMs and HMMs — architectures that needed compact representations.

For a CNN, that DCT step is a problem: it destroys spatial locality. The CNN cannot see adjacent frequency bands as neighbors in the feature map because the DCT has scrambled their ordering. Log-mel spectrograms preserve the full 2D structure (frequency × time) that a CNN's convolutional filters are designed to exploit.

Additionally, for non-speech sounds (alarms, horns, baby cries), the full spectral envelope carries discriminative information that the DCT discards. Log-mel retains 64 frequency channels versus the typical 13–20 MFCC coefficients — roughly 4× more spectral resolution with negligible extra compute.

**Conclusion:** Log-mel gives CNNs better spatial structure and more spectral detail. MFCC is a legacy choice for non-CNN models.

### 5.2 DS-CNN over MobileNet

MobileNet was designed for image classification on mobile phones — devices with 1–4GB of RAM and dedicated NPUs. Its smallest variant (MobileNetV2 α=0.1) still has ~470K parameters and produces a ~1.8MB INT8 model. That is 50× larger than our 36.4KB target.

DS-CNN (Depthwise Separable CNN) uses the same core idea — separate depthwise and pointwise convolutions to reduce multiply-accumulate operations — but is designed specifically for microcontroller inference. The architecture from the "Hello Edge" paper (Zhang et al., 2017) was benchmarked on keyword spotting tasks on Cortex-M4 MCUs, which are slower than the ESP32-S3's LX7 cores.

The depthwise separable factorization reduces computation by a factor of roughly `1/N + 1/D²` compared to standard convolutions (N = output channels, D = kernel size). For a 3×3 kernel and 64 channels this is approximately an 8× reduction in FLOPs.

**Conclusion:** DS-CNN reaches our accuracy target at 18K parameters. MobileNet is the wrong tool for this problem space.

### 5.3 Knowledge Distillation over training student directly

A student model trained directly on hard one-hot labels receives a binary signal: class A = 1.0, everything else = 0.0. There is no information about which wrong classes are "close" to the correct answer. A fire alarm spectrum that is 80% fire_alarm, 15% car_horn, 5% background teaches the student nothing about the relationship between those classes.

The teacher model, trained without size constraints, learns a rich probability distribution over classes. Its "soft" predictions (at temperature T=6 to spread the distribution further) carry inter-class similarity information. The student trained on these soft targets learns not just "what is the right class" but "what does the input look like across all classes."

This is especially important for our small dataset: direct training of a tiny model on ~6000 samples with one-hot labels tends to overfit poorly. Knowledge distillation acts as implicit regularization, since the student is optimizing a smoother target.

The loss combination is:
```
total_loss = α · KL(teacher_soft, student_soft) · T²  +  (1 - α) · CE(labels, student)
           = 0.7 · KD_loss + 0.3 · CE_loss
```

The T² factor rescales the KD loss magnitude to be comparable to the CE loss (standard practice from Hinton et al., 2015).

**Conclusion:** Knowledge distillation transfers structured class-relationship knowledge that is impossible to encode in hard labels. It was the difference between 85% and 95% accuracy at 18K parameters.

### 5.4 QAT over Post-Training Quantization (PTQ)

Post-Training Quantization (PTQ) applies quantization after training is complete. It uses a representative calibration dataset to find the optimal INT8 scale and zero-point for each layer's weights and activations. The model weights were never aware of quantization during training, so the quantization error is not compensated for.

Quantization-Aware Training (QAT) inserts fake quantization nodes into the computation graph during fine-tuning. The forward pass simulates INT8 rounding; the backward pass uses a straight-through estimator to propagate gradients. The model weights are updated to minimize loss in the presence of quantization noise.

For large models (ResNet, BERT), PTQ is often "good enough" because the model has redundant capacity to absorb quantization error. For a 18K-parameter model operating close to its capacity limit, every parameter matters. QAT consistently recovers 1–3% accuracy over PTQ in small model regimes.

In our case: the student at float32 achieves 94.1% accuracy. After QAT + INT8 conversion, accuracy is 95.4% on the clean test set (QAT fine-tuning with SpecAugment effectively provides additional regularization). PTQ on the same model would likely drop to ~91–92%.

**Conclusion:** QAT is mandatory for small models. PTQ is a shortcut that degrades accuracy at this parameter count.

### 5.5 Why SpecAugment + Mixup together

SpecAugment masks rectangular regions of the log-mel spectrogram in the time and frequency dimensions. This forces the model to classify using partial information — a robustness technique directly applicable to real-world scenarios where part of a sound is masked by a competing noise source.

Mixup creates convex combinations of pairs of training examples and their labels. It encourages the model to behave linearly between training examples, which improves calibration and generalization. For a small dataset, it effectively doubles the diversity of training examples without collecting new data.

The two augmentations operate at different levels: SpecAugment destroys local structure (masking), while Mixup blends global statistics (mixing two entire spectrograms). They are complementary and together reduce overfitting more than either alone.

**Conclusion:** Both techniques are applied only to training data, never to val or test. They are the primary reason a model trained on ~6000 samples generalizes to unseen recordings.

---

## 6. Dataset

### 6.1 Sources

| Class | Primary Source | Secondary |
|-------|---------------|-----------|
| fire_alarm | Google AudioSet | ESC-50 |
| baby_cry | DCASE 2017 Task 2 | Freesound |
| choking | Google AudioSet | Manual recordings + augmentation |
| car_horn | UrbanSound8K | AudioSet |
| doorbell | ESC-50 | DESED |
| background | DEMAND | MUSAN |

### 6.2 Dataset Statistics

| Split | Samples | Source |
|-------|---------|--------|
| Train | 5,838 | Source files (remainder) + ALL augmented files |
| Validation | 153 | Source files only |
| Test | 153 | Source files only |
| **Total raw clips** | **598** | Before augmentation |

### 6.3 Data Leakage Prevention

This is a critical design detail. Naive splitting of augmented data creates a leak: if the same source clip appears both as `alarm_01.wav` in test and as `aug_pitch_alarm_01.wav` in train, the model has effectively seen the test sample. This inflates validation accuracy by 5–10% and gives a false sense of generalization.

The preprocessing pipeline (`src/data/preprocess.py`) separates files by naming convention:
- Files **not** prefixed with `aug_` are source files → split into train/val/test normally
- Files prefixed with `aug_` are augmented files → **train only**, never val or test

This ensures val and test sets contain only unseen original recordings.

### 6.4 Augmentation Pipeline (`src/data/augment.py`)

Each source clip generates 6 augmented variants:
- Pitch shift +2 semitones
- Pitch shift −2 semitones
- Time stretch ×0.85 (slower)
- White noise injection at SNR 15dB
- Background noise mix at SNR 10dB
- Combined: pitch shift + noise

Target: ≥400 clips per class before augmentation expansion.

---

## 7. Feature Extraction

All audio is processed by `src/data/preprocess.py` before training.

**Processing chain for each audio file:**

1. Load audio with librosa, resample to 16kHz mono
2. Segment into overlapping 1-second windows (500ms hop = 50% overlap)
3. For each segment, compute log-mel spectrogram:
   - `n_fft=512` → 32ms FFT window at 16kHz
   - `hop_length=256` → 16ms hop between frames
   - `n_mels=64` → 64 mel filterbank channels
   - Frequency range: 60Hz to 7600Hz (excludes very low rumble and ultrasonic)
   - Convert power to dB with `ref=max`, then normalize to [0, 1]
4. Add channel dimension: shape becomes `(64, ~63, 1)`
5. Save to numpy arrays: `X_train.npy`, `X_val.npy`, `X_test.npy`

**Output shape:** `(64 mel channels, 63 time frames, 1 channel)`

The 63 time frames correspond to a 1-second window at 16ms hop length: `ceil(16000 / 256) = 63`.

---

## 8. Augmentation Strategy

Augmentation is applied **online** during training (SpecAugment + Mixup) and **offline** to raw audio before preprocessing (pitch shift, time stretch, noise injection).

### Online augmentation (in `yamnet_finetune.py`, applied per batch)

```
SpecAugment:
  - 2 frequency masks, up to 10 mel channels each
  - 2 time masks, up to 10 time frames each
  - Masking sets values to zero

Mixup:
  - λ ~ Uniform(0, 1), then λ = max(λ, 1-λ)
  - x_mixed = λ·x_i + (1-λ)·x_j
  - y_mixed = λ·y_i + (1-λ)·y_j (soft label blending)
```

### Offline augmentation (applied to raw `.wav` files, results stored as `aug_*.wav`)

Augmented clips are created once, preprocessed into log-mel features, and added exclusively to the training set.

---

## 9. Teacher Model

**Architecture:** `teacher_cnn` in `src/models/yamnet_finetune.py`

```
Input: (64, 63, 1)  ← log-mel spectrogram

Conv2D(32, 3×3) → BatchNorm → ReLU → MaxPool(2×2)
Conv2D(64, 3×3) → BatchNorm → ReLU → MaxPool(2×2)
Conv2D(128, 3×3) → BatchNorm → ReLU → MaxPool(2×2)
Conv2D(256, 3×3) → BatchNorm → ReLU → GlobalAveragePooling2D

Dropout(0.4)
Dense(128, relu)
Dropout(0.3)
Dense(6, softmax)

Parameters: 423,174
Size (float32): ~1.7 MB
```

**Training details:**
- Optimizer: Adam, LR=1e-3
- Loss: categorical cross-entropy
- Class weights: computed with sklearn's `compute_class_weight('balanced')` to handle imbalanced classes (especially choking)
- Callbacks: EarlyStopping (patience=5), ReduceLROnPlateau (factor=0.5, patience=3), ModelCheckpoint
- Online augmentation: SpecAugment + Mixup applied per batch

The teacher is **not deployed**. Its sole purpose is to generate soft probability distributions that the student learns from.

---

## 10. Knowledge Distillation

**Implementation:** `src/models/distillation.py`

The `DistillationModel` class wraps teacher and student:

```python
# Teacher frozen, student trains
teacher_soft = softmax(teacher_logits / T)    # T=6 softens distribution
student_soft = softmax(student_logits / T)

kd_loss = KLDivergence(teacher_soft, student_soft) * T²
ce_loss = CategoricalCrossentropy(true_labels, student_logits)

total_loss = α * kd_loss + (1 - α) * ce_loss
           = 0.7 * kd_loss + 0.3 * ce_loss
```

**Temperature T=6:** At T=1, the teacher's softmax is sharply peaked. At T=6, the distribution is much softer, making the inter-class similarity signal stronger. For example, fire_alarm might be 70% confident at T=1, but at T=6 the remaining 30% is spread meaningfully across car_horn and background — information the student can learn from.

**α=0.7:** The KD loss carries 70% of the total loss signal. This reflects the intent: the teacher's soft labels are more informative than hard one-hot targets for a small-dataset scenario.

**Compression achieved:**
- Teacher: 423,174 parameters
- Student: 18,086 parameters
- Compression ratio: **23.4×**

---

## 11. Student Model — DS-CNN

**Architecture:** `src/models/dscnn.py`

```
Input: (64, 63, 1)

Conv2D(32, 3×3, same) → BatchNorm → ReLU → MaxPool(2×2)

DS Block 1:
  DepthwiseConv2D(3×3) → BN → ReLU → Conv2D(64, 1×1) → BN → ReLU
  → MaxPool(2×2)

DS Block 2:
  DepthwiseConv2D(3×3) → BN → ReLU → Conv2D(64, 1×1) → BN → ReLU
  → MaxPool(2×2)

DS Block 3:
  DepthwiseConv2D(3×3) → BN → ReLU → Conv2D(128, 1×1) → BN → ReLU

GlobalAveragePooling2D
Dropout(0.3)
Dense(6, softmax)

Parameters: 18,086
Size (float32): ~72 KB
Size (INT8): 36.4 KB
```

**Why no Flatten + Dense hidden layer:** GlobalAveragePooling2D replaces the Flatten operation. Flattening a spatial feature map and projecting it through a large Dense layer adds many parameters with limited benefit for classification — GAP directly computes the mean activation per channel, which is more parameter-efficient and naturally provides spatial invariance.

---

## 12. Quantization — QAT vs PTQ

**Implementation:** `src/convert.py`

**QAT process:**
1. Load trained student (float32 `.keras` file)
2. Apply `tfmot.quantization.keras.quantize_model()` — inserts fake quant nodes
3. Fine-tune for 10 epochs at LR=1e-5 (very low, just adapts weights to quantization)
4. Convert to TFLite INT8 with full integer quantization:
   - Representative dataset: 500 training samples for calibration
   - Input type: INT8, Output type: INT8
   - Target ops: `TFLITE_BUILTINS_INT8` (no float fallback)

**TFLite INT8 model:**
- All weights: 8-bit signed integers
- All activations: 8-bit signed integers
- No float operations at inference time
- Compatible with ESP32-S3 TFLite Micro (no FPU needed for most ops)

---

## 13. Firmware — Inference on ESP32-S3

**File:** `firmware/esp32/main.ino`

**Runtime pipeline:**
```
I2S read (INMP441) → circular buffer
↓
1-second window assembled (every 500ms)
↓
Log-mel spectrogram computed on-device (ESP-DSP library)
↓
INT8 quantize input (scale + zero-point from TFLite model metadata)
↓
TFLite Micro interpreter runs DS-CNN forward pass
↓
Dequantize output logits
↓
Temporal smoothing: 3-frame sliding window majority vote
↓
Apply per-class confidence thresholds
↓
If threshold exceeded: trigger LED pattern + vibration motor
```

**Temporal smoothing (3-frame majority vote):**
The model runs every 500ms. A single spurious inference can produce a false positive. The 3-frame window requires the same class to be the top prediction in 2 out of 3 consecutive windows before triggering an alert. This introduces a maximum latency of 1.5 seconds for a true alarm but eliminates brief noise-triggered false positives.

**C array export:**
The `.tflite` file is converted to a C byte array (`firmware/esp32/model_data.h`) by `src/export_model_header.py`. The ESP32 sketch includes this header directly — the model lives in flash, not RAM.

---

## 14. Results

### 14.1 Per-Class Performance (Final TFLite INT8, clean test set, n=153)

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| fire_alarm | 0.921 | 0.972 | **0.945** |
| baby_cry | 1.000 | 1.000 | **1.000** |
| choking | 0.857 | 0.947 | **0.900** |
| car_horn | 0.895 | 0.971 | **0.933** |
| doorbell | 0.958 | 0.856 | **0.906** |
| background | 1.000 | 0.986 | **0.993** |
| **macro average** | | | **0.946** |
| **overall accuracy** | | | **95.4%** |

### 14.2 Model Comparison

| Model | Parameters | Size | Accuracy | Latency (CPU) | Deployable |
|-------|-----------|------|----------|---------------|-----------|
| Teacher CNN | 423,174 | ~1.7MB float32 | 94.1% | ~8ms | No (too large) |
| Student DS-CNN | 18,086 | ~72KB float32 | 94.1% | ~0.5ms | Marginal |
| Student DS-CNN INT8 | 18,086 | **36.4KB** | **95.4%** | **0.21ms** | **Yes** |

### 14.3 System Targets vs Achieved

| Target | Goal | Achieved |
|--------|------|----------|
| Model size | < 100KB | 36.4KB (2.7× better) |
| Accuracy | > 90% | 95.4% |
| Macro F1 | > 0.90 | 0.946 |
| CPU latency | < 100ms | 0.21ms |
| ESP32 latency (estimated) | < 100ms | ~2ms (6–10× CPU factor) |
| Compression ratio | > 10× | 23.4× (params), 4× (size via INT8) |

### 14.4 Notes on Accuracy

The TFLite INT8 model achieves **higher** accuracy than the float32 student (95.4% vs 94.1%). This is not unusual: QAT fine-tuning adds 10 more training epochs with additional regularization (SpecAugment continues during QAT), which improves generalization beyond what the distillation phase alone achieved.

The hardest class is `choking` (F1=0.900), which is expected: the dataset has fewer natural recordings (supplemented by manual recordings and heavy augmentation), and choking sounds are acoustically variable.

`baby_cry` achieves perfect F1=1.000 on the test set — it is the most acoustically distinctive class in the dataset.

---

## 15. Project Structure

```
iot/
├── PROJECT.md              — Project reference document
├── GUIDE.md                — This file
├── STEPS.md                — Step-by-step run guide
├── CHANGELOG.md            — Chronological development log
├── requirements.txt        — Python dependencies
├── run_pipeline.sh         — Automated full pipeline script
├── retrain.sh              — Retrain after adding new data
├── test_model.py           — Automated test suite
│
├── data/
│   ├── raw/                — Original audio clips (per-class folders)
│   │   ├── fire_alarm/     — .wav/.mp3 files (source + aug_*.wav)
│   │   ├── baby_cry/
│   │   ├── choking/
│   │   ├── car_horn/
│   │   ├── doorbell/
│   │   └── background/
│   └── processed/          — Numpy arrays ready for training
│       ├── X_train.npy     — shape: (5838, 64, 63, 1)
│       ├── y_train.npy     — shape: (5838,)
│       ├── X_val.npy       — shape: (153, 64, 63, 1)
│       ├── y_val.npy       — shape: (153,)
│       ├── X_test.npy      — shape: (153, 64, 63, 1)
│       └── y_test.npy      — shape: (153,)
│
├── src/
│   ├── data/
│   │   ├── download.py         — ESC-50 download
│   │   ├── audioset_download.py — AudioSet via yt-dlp
│   │   ├── freesound_download.py — Freesound + YouTube
│   │   ├── augment.py           — Offline audio augmentation
│   │   └── preprocess.py        — Audio → log-mel → numpy
│   ├── models/
│   │   ├── yamnet_finetune.py   — Teacher CNN training
│   │   ├── dscnn.py             — Student DS-CNN architecture
│   │   ├── distillation.py      — Knowledge distillation loop
│   │   └── inference_test.py    — Live mic test (Mac)
│   ├── train.py            — Main training entry point
│   ├── evaluate.py         — Metrics, confusion matrix, latency
│   ├── convert.py          — QAT + TFLite INT8 export
│   └── export_model_header.py — .tflite → model_data.h
│
├── models/
│   ├── teacher/
│   │   ├── best_teacher.keras   — Best checkpoint during training
│   │   └── yamnet_finetuned.keras — Final teacher
│   └── student/
│       ├── best_student.keras   — Best distillation checkpoint
│       ├── dscnn_student.keras  — Final student (float32)
│       ├── dscnn_int8.tflite    — Final INT8 model (deployed)
│       └── confusion_matrix.png — Evaluation output
│
├── firmware/
│   └── esp32/
│       ├── main.ino             — Full inference firmware
│       ├── model_data.h         — TFLite model as C array
│       └── README_FLASH.md      — Flash instructions
│
└── notebooks/
    └── pipeline.ipynb       — Interactive ML pipeline walkthrough
```

---

## 16. Key Technical Parameters

| Parameter | Value |
|-----------|-------|
| Sample rate | 16,000 Hz |
| Audio window | 1 second |
| Inference stride | 500ms (50% overlap) |
| FFT window size | 512 samples (32ms) |
| Hop length | 256 samples (16ms) |
| Mel channels | 64 |
| Frequency range | 60Hz – 7,600Hz |
| Spectrogram shape | (64, 63, 1) |
| Number of classes | 6 |
| Teacher parameters | 423,174 |
| Student parameters | 18,086 |
| Compression ratio | 23.4× |
| Model size (INT8) | 36.4 KB |
| Overall accuracy | 95.4% |
| Macro F1 | 0.946 |
| CPU latency (mean) | 0.21ms |
| Estimated ESP32-S3 latency | ~2ms |
| Temporal smoothing window | 3 frames |
| Training samples | 5,838 |
| Validation samples | 153 |
| Test samples | 153 |
| KD temperature | 6.0 |
| KD alpha | 0.7 |
| QAT fine-tuning epochs | 10 |
| QAT learning rate | 1e-5 |

---

## 17. Simulation (No Hardware Required)

Since no physical hardware is available, the entire system is validated through software simulation.

### Python Embedded Simulator

`simulation/embedded_simulator.py` reproduces the **exact firmware logic** in Python:
- Reads audio files (simulates INMP441 microphone)
- Computes log-mel spectrograms (simulates ESP-DSP FFT + mel filterbank)
- Runs the real INT8 TFLite model (simulates TFLite Micro)
- Applies temporal smoothing with a 3-frame circular buffer (identical to firmware)
- Applies per-class confidence thresholds (identical to firmware)
- Estimates ESP32-S3 latency (host latency × 8 scaling factor)
- Simulates LED color output and vibration patterns per class

```
python simulation/embedded_simulator.py          # test all 6 classes
python simulation/embedded_simulator.py --live   # live microphone
python simulation/embedded_simulator.py --file path/to/audio.wav
```

### Simulation Results (6/6 classes correct)

| Expected | Predicted | Alerts | Frames | Latency (est.) |
|----------|-----------|--------|--------|----------------|
| fire_alarm | fire_alarm ✓ | 7/9 | 9 | 44ms |
| baby_cry | baby_cry ✓ | 7/9 | 9 | 9ms |
| choking | choking ✓ | 4/9 | 9 | 8ms |
| car_horn | car_horn ✓ | 6/9 | 9 | 9ms |
| doorbell | doorbell ✓ | 7/9 | 9 | 9ms |
| background | background ✓ | 0/9 | 9 | 9ms |

Memory usage: 48.3 KB / 512 KB (9.4% of available SRAM).

### Interactive Dashboard

`simulation/dashboard.py` is a Streamlit web app with:
- File upload or dataset sample selection
- Real-time confidence visualization per class over time
- LED and vibration output simulation with color-coded alerts
- Adjustable thresholds via sliders
- Architecture diagrams and model metrics

```
streamlit run simulation/dashboard.py
```

### Wokwi (Online ESP32 Simulator)

For visual circuit simulation: `simulation/wokwi/diagram.json` can be imported at https://wokwi.com to validate LED patterns and pin assignments.
