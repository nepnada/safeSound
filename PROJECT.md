# Sound Alert System for the Hearing-Impaired
## Project Reference Document

---

## Context

Wearable IoT device for deaf/hard-of-hearing users. The device continuously listens to ambient audio, classifies critical sounds using an on-device ML model, and alerts the user via RGB LEDs and vibration motor.

**Hardware:**
- Microcontroller: ESP32-S3 (dual-core LX7 @ 240MHz, 8MB PSRAM)
- Microphone: INMP441 (I2S digital, 16kHz, SNR 61dB)
- Output: WS2812B RGB LEDs + ERM vibration motor
- Power: LiPo 3.7V 1000mAh + TP4056

**6 sound classes:**

| Class | Priority |
|-------|----------|
| Fire / smoke alarm | Critical |
| Baby crying | High |
| Person choking | Critical |
| Car horn | High |
| Doorbell | Medium |
| Background noise | — (reject class) |

---

## ML Pipeline (Final — Do Not Change)

```
16kHz audio (INMP441)
    ↓
Log-Mel Spectrogram — 64 channels, 32ms window, 16ms hop
    ↓
SpecAugment + Mixup  ← augmentation
    ↓
YAMNet fine-tuning on 6 classes  ← transfer learning (teacher model)
    ↓
Knowledge Distillation → DS-CNN (~60KB)  ← student model for ESP32
    ↓
QAT INT8 from the start
    ↓
Temporal smoothing (3-frame window) + per-class confidence thresholds
    ↓
TFLite Micro → ESP32-S3
```

### Why each choice

| Choice | Reason |
|--------|--------|
| Log-Mel 64ch | Better than MFCC for CNN, captures more frequency detail |
| SpecAugment + Mixup | Handles small datasets, reduces overfitting |
| YAMNet fine-tune | Pre-trained on 521 AudioSet classes, strong audio features |
| Knowledge Distillation | Compresses YAMNet (~4MB) into DS-CNN (~60KB) for ESP32 |
| QAT from start | Better INT8 accuracy than post-training quantization |
| Temporal smoothing | Reduces false positives on streaming audio |
| DS-CNN | Depthwise separable CNN, designed for MCU inference |

---

## Datasets

| Class | Source |
|-------|--------|
| Fire / smoke alarm | Google AudioSet + ESC-50 |
| Baby crying | DCASE 2017 Task 2 + Freesound |
| Person choking | Google AudioSet + manual recordings + augmentation |
| Car horn | UrbanSound8K |
| Doorbell | ESC-50 + DESED |
| Background noise | DEMAND + Musan |

**Target:** ~500–1000 clips per class, 1s segments, 16kHz mono

---

## 14-Day Plan

### Week 1 — Working pipeline end-to-end

| Day | Task | Output |
|-----|------|--------|
| J1 | Project structure + environment setup + dataset download scripts | `data/raw/` populated |
| J2 | Preprocessing pipeline: resample → log-mel → segment → export | `data/processed/train.npy`, `test.npy` |
| J3 | YAMNet fine-tuning script + SpecAugment + Mixup | Trained teacher model |
| J4 | DS-CNN architecture + Knowledge Distillation training loop + QAT | Student model `.tflite` <100KB |
| J5 | Flash ESP32-S3 + real microphone test + end-to-end validation | **System running on device** |

### Week 2 — Improvements without rebuilding

| Day | Task | Output |
|-----|------|--------|
| J6 | Data audit: balance classes, clean bad samples, add RIR augmentation | Better training set |
| J7 | Retrain student model with improved data (same architecture, same code) | +3–5% accuracy |
| J8 | Temporal smoothing + per-class thresholds (fire=0.75, doorbell=0.90) | −50% false positives |
| J9 | BLE notification to smartphone (already on ESP32-S3) | Extra feature |
| J10 | Final tests: latency measurement, confusion matrix, F1 per class | Complete evaluation |

---

## Project Structure

```
iot/
├── PROJECT.md              ← this file (reference doc)
├── GUIDE.md                ← complete project guide (architecture, choices, results)
├── STEPS.md                ← step-by-step run guide (bash commands)
├── CHANGELOG.md            ← change log
├── data/
│   ├── raw/                ← downloaded audio clips
│   └── processed/          ← numpy arrays ready for training
├── src/
│   ├── data/
│   │   ├── download.py     ← dataset download scripts
│   │   ├── preprocess.py   ← audio → log-mel → numpy
│   │   └── augment.py      ← ×6 augmentation pipeline
│   ├── models/
│   │   ├── yamnet_finetune.py   ← teacher model
│   │   ├── dscnn.py             ← student model architecture
│   │   ├── distillation.py     ← KD training loop
│   │   └── inference_test.py   ← live microphone test
│   ├── train.py            ← main training entry point
│   ├── evaluate.py         ← confusion matrix, F1, latency
│   ├── convert.py          ← TFLite INT8 export
│   └── export_model_header.py  ← .tflite → C header
├── simulation/
│   ├── embedded_simulator.py   ← full ESP32 pipeline simulation
│   ├── dashboard.py            ← Streamlit interactive dashboard
│   ├── results/                ← JSON simulation reports
│   └── wokwi/                  ← Wokwi circuit diagram
├── firmware/
│   └── esp32/              ← Arduino/ESP-IDF code + model_data.h
├── notebooks/
│   ├── pipeline.ipynb      ← documented ML pipeline notebook
│   └── figures/            ← generated visualizations
├── models/
│   ├── teacher/            ← trained teacher weights
│   └── student/            ← DS-CNN .tflite final model
├── test_model.py           ← automated test suite
├── run_pipeline.sh         ← one-command full pipeline
├── retrain.sh              ← retrain after adding data
└── requirements.txt
```

---

## Key Technical Specs

| Parameter | Value |
|-----------|-------|
| Sample rate | 16 kHz |
| Audio window | 1 second |
| Inference stride | 500ms (50% overlap) |
| Log-mel channels | 64 |
| FFT window | 32ms |
| Hop length | 16ms |
| Model size target | < 100KB INT8 |
| Latency target | < 100ms per window |
| Confidence threshold (fire) | 0.75 |
| Confidence threshold (doorbell) | 0.90 |
| Temporal smoothing | 3-frame majority vote |

---

## Constraints

- Model must run on ESP32-S3 with TFLite Micro
- No cloud dependency at inference time
- Total inference latency < 500ms end-to-end
- Model size < 100KB (INT8 quantized)
