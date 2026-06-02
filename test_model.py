"""
Quick test script — run after training to validate the full pipeline.
Tests: model loads, input shape, inference runs, output is valid probabilities.

Usage: python test_model.py
"""

import sys
import numpy as np
import tensorflow as tf
from pathlib import Path

CLASSES = ["fire_alarm", "baby_cry", "choking", "car_horn", "doorbell", "background"]
TFLITE_PATH = "models/student/dscnn_int8.tflite"
KERAS_PATH  = "models/student/dscnn_student.keras"

def green(s): return f"\033[92m{s}\033[0m"
def red(s):   return f"\033[91m{s}\033[0m"
def bold(s):  return f"\033[1m{s}\033[0m"

def check(label, ok, detail=""):
    mark = green("✓") if ok else red("✗")
    print(f"  {mark}  {label}" + (f"  — {detail}" if detail else ""))
    return ok

def test_preprocessing():
    print(bold("\n[1] Preprocessing pipeline"))
    try:
        import librosa
        from src.data.preprocess import audio_to_logmel, segment_audio
        audio = np.random.randn(16000).astype(np.float32)
        segs = segment_audio(audio)
        lm = audio_to_logmel(segs[0])
        check("audio_to_logmel runs", True, f"shape {lm.shape}")
        check("shape is (64, 63)", lm.shape == (64, 63), str(lm.shape))
        check("values in [0, 1]", 0 <= lm.min() and lm.max() <= 1, f"min={lm.min():.3f} max={lm.max():.3f}")
    except Exception as e:
        check("preprocessing", False, str(e))

def test_keras_model():
    print(bold("\n[2] Keras student model"))
    if not Path(KERAS_PATH).exists():
        check("model file exists", False, f"{KERAS_PATH} not found — run training first")
        return
    try:
        model = tf.keras.models.load_model(KERAS_PATH)
        check("model loads", True, f"{model.count_params():,} params")
        x = np.random.rand(1, 64, 63, 1).astype(np.float32)
        pred = model(x, training=False).numpy()
        check("inference runs", True, f"output shape {pred.shape}")
        check("output sums to ~1", abs(pred.sum() - 1.0) < 0.01, f"sum={pred.sum():.4f}")
        check("6 classes", pred.shape[1] == 6, str(pred.shape))
        best = CLASSES[np.argmax(pred)]
        check("prediction valid", True, f"argmax → {best} ({pred.max():.3f})")
    except Exception as e:
        check("model test", False, str(e))

def test_tflite_model():
    print(bold("\n[3] TFLite INT8 model"))
    if not Path(TFLITE_PATH).exists():
        check("tflite file exists", False, f"{TFLITE_PATH} not found — run conversion first")
        return
    try:
        size_kb = Path(TFLITE_PATH).stat().st_size / 1024
        check("file size < 100KB", size_kb < 100, f"{size_kb:.1f} KB")

        interp = tf.lite.Interpreter(model_path=TFLITE_PATH)
        interp.allocate_tensors()
        inp = interp.get_input_details()[0]
        out = interp.get_output_details()[0]

        check("input dtype INT8", inp["dtype"] == np.int8, str(inp["dtype"]))
        check("output dtype INT8", out["dtype"] == np.int8, str(out["dtype"]))

        scale_in, zero_in = inp["quantization"]
        scale_out, zero_out = out["quantization"]

        x = np.random.rand(1, 64, 63, 1).astype(np.float32)
        x_q = (x / scale_in + zero_in).astype(np.int8)
        interp.set_tensor(inp["index"], x_q)
        interp.invoke()
        y_q = interp.get_tensor(out["index"])
        probs = (y_q.astype(np.float32) - zero_out) * scale_out

        check("inference runs", True)
        check("6 output classes", probs.shape[1] == 6, str(probs.shape))
        best = CLASSES[np.argmax(probs)]
        check("valid prediction", True, f"argmax → {best} ({probs.max():.3f})")

    except Exception as e:
        check("tflite test", False, str(e))

def test_dataset():
    print(bold("\n[4] Dataset"))
    try:
        X_train = np.load("data/processed/X_train.npy")
        y_train = np.load("data/processed/y_train.npy")
        X_val   = np.load("data/processed/X_val.npy")

        check("files exist", True)
        check("input shape correct", X_train.shape[1:] == (64, 63, 1), str(X_train.shape))
        check("labels in range", y_train.max() <= 5 and y_train.min() >= 0,
              f"min={y_train.min()} max={y_train.max()}")
        from collections import Counter
        dist = Counter(y_train)
        min_cls = min(dist.values())
        check("all classes present", len(dist) == 6, str(dict(dist)))
        check("min class >= 100 segments", min_cls >= 100, f"min={min_cls}")
        print(f"\n    Train: {len(X_train)} | Val: {len(X_val)}")
        for i, name in enumerate(CLASSES):
            print(f"    {name:15s}: {dist.get(i,0):4d}")
    except Exception as e:
        check("dataset", False, str(e))

if __name__ == "__main__":
    print(bold("="*50))
    print(bold(" Sound Alert System — Test Suite"))
    print(bold("="*50))

    test_preprocessing()
    test_dataset()
    test_keras_model()
    test_tflite_model()

    print(bold("\n" + "="*50))
    print("Run 'python src/evaluate.py' for full metrics\n")
