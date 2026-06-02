"""
Evaluation: confusion matrix, F1 per class, latency benchmark.
Run after training to get full metrics.

Usage: python src/evaluate.py --model models/student/dscnn_int8.tflite
"""

import numpy as np
import tensorflow as tf
import argparse
import time
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")  # No display required

CLASSES = ["fire_alarm", "baby_cry", "choking", "car_horn", "doorbell", "background"]
DATA_DIR = Path("data/processed")


def evaluate_tflite(model_path: str):
    X_test = np.load(DATA_DIR / "X_test.npy")
    y_test = np.load(DATA_DIR / "y_test.npy")

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]

    scale_in, zero_in = inp["quantization"]
    scale_out, zero_out = out["quantization"]

    preds = []
    latencies = []

    for i in range(len(X_test)):
        x = X_test[i:i+1].astype(np.float32)
        x_q = (x / scale_in + zero_in).astype(np.int8)

        t0 = time.perf_counter()
        interpreter.set_tensor(inp["index"], x_q)
        interpreter.invoke()
        latencies.append((time.perf_counter() - t0) * 1000)

        y_q = interpreter.get_tensor(out["index"])
        pred = np.argmax((y_q.astype(np.float32) - zero_out) * scale_out)
        preds.append(pred)

    preds = np.array(preds)

    print("\n=== Classification Report ===")
    print(classification_report(y_test, preds, target_names=CLASSES, digits=3))

    print(f"\n=== Latency (on host CPU, ~6-10x faster than ESP32) ===")
    print(f"  Mean: {np.mean(latencies):.2f}ms | P95: {np.percentile(latencies, 95):.2f}ms")

    # Confusion matrix
    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASSES)))
    ax.set_yticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES, rotation=45, ha="right")
    ax.set_yticklabels(CLASSES)
    plt.colorbar(im)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    out = Path(model_path).parent / "confusion_matrix.png"
    plt.savefig(out, dpi=150)
    print(f"\nConfusion matrix saved → {out}")


def evaluate_keras(model_path: str):
    X_test = np.load(DATA_DIR / "X_test.npy")
    y_test = np.load(DATA_DIR / "y_test.npy")

    model = tf.keras.models.load_model(model_path)
    preds = np.argmax(model.predict(X_test, batch_size=32), axis=1)

    print("\n=== Classification Report ===")
    print(classification_report(y_test, preds, target_names=CLASSES, digits=3))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/student/dscnn_int8.tflite")
    args = parser.parse_args()

    if args.model.endswith(".tflite"):
        evaluate_tflite(args.model)
    else:
        evaluate_keras(args.model)
