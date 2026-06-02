"""
J4/J5 — QAT + TFLite INT8 conversion.

Applies Quantization-Aware Training (QAT) then converts to TFLite INT8.
QAT simulates quantization noise during training → better INT8 accuracy.

Output: models/student/dscnn_int8.tflite
"""

import numpy as np
import tensorflow as tf
from pathlib import Path

try:
    import tensorflow_model_optimization as tfmot
    QAT_AVAILABLE = True
except ImportError:
    QAT_AVAILABLE = False
    print("[WARN] tensorflow-model-optimization not installed. Using PTQ fallback.")

DATA_DIR = Path("data/processed")
STUDENT_PATH = Path("models/student/dscnn_student.keras")
OUT_DIR = Path("models/student")

BATCH_SIZE = 32
QAT_EPOCHS = 10
QAT_LR = 1e-5


def apply_qat_and_convert():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    X_train = np.load(DATA_DIR / "X_train.npy")
    y_train = np.load(DATA_DIR / "y_train.npy")
    X_val = np.load(DATA_DIR / "X_val.npy")
    y_val = np.load(DATA_DIR / "y_val.npy")

    y_train_oh = tf.keras.utils.to_categorical(y_train, 6)
    y_val_oh = tf.keras.utils.to_categorical(y_val, 6)

    # Load student
    model = tf.keras.models.load_model(str(STUDENT_PATH))
    print(f"Loaded student: {model.count_params():,} params")

    if QAT_AVAILABLE:
        print("\n=== Applying QAT ===")
        qat_model = tfmot.quantization.keras.quantize_model(model)
        qat_model.compile(
            optimizer=tf.keras.optimizers.Adam(QAT_LR),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        train_ds = (
            tf.data.Dataset.from_tensor_slices((X_train, y_train_oh))
            .shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        )
        val_ds = (
            tf.data.Dataset.from_tensor_slices((X_val, y_val_oh))
            .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        )

        qat_model.fit(train_ds, validation_data=val_ds, epochs=QAT_EPOCHS,
                      callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)])

        convert_model = qat_model
    else:
        convert_model = model

    # ── TFLite INT8 conversion ───────────────────────────────────────────────
    print("\n=== Converting to TFLite INT8 ===")

    def representative_dataset():
        for i in range(0, min(500, len(X_train)), BATCH_SIZE):
            batch = X_train[i:i+BATCH_SIZE].astype(np.float32)
            yield [batch]

    converter = tf.lite.TFLiteConverter.from_keras_model(convert_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    out_path = OUT_DIR / "dscnn_int8.tflite"
    with open(out_path, "wb") as f:
        f.write(tflite_model)

    size_kb = len(tflite_model) / 1024
    print(f"TFLite INT8 model: {size_kb:.1f} KB → {out_path}")

    # ── Verify TFLite accuracy ───────────────────────────────────────────────
    print("\n=== Verifying TFLite accuracy ===")
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]

    scale_in, zero_in = inp["quantization"]
    scale_out, zero_out = out["quantization"]

    correct = 0
    total = min(200, len(X_val))
    for i in range(total):
        x = X_val[i:i+1].astype(np.float32)
        x_q = (x / scale_in + zero_in).astype(np.int8)
        interpreter.set_tensor(inp["index"], x_q)
        interpreter.invoke()
        y_q = interpreter.get_tensor(out["index"])
        pred = np.argmax((y_q.astype(np.float32) - zero_out) * scale_out)
        if pred == y_val[i]:
            correct += 1

    print(f"TFLite accuracy (sample): {correct/total*100:.1f}%")
    return out_path


if __name__ == "__main__":
    apply_qat_and_convert()
