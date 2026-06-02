"""
J4 — Knowledge Distillation: Teacher CNN → Student DS-CNN.

The student (DS-CNN) learns from:
  - Hard labels (ground truth) → cross-entropy
  - Soft labels (teacher predictions) → KL divergence at temperature T

Loss = α * KL(teacher_soft, student_soft) + (1 - α) * CE(labels, student)
"""

import numpy as np
import tensorflow as tf
from pathlib import Path
from src.models.dscnn import build_dscnn

# ── Config ────────────────────────────────────────────────────────────────────
NUM_CLASSES = 6
BATCH_SIZE = 32
EPOCHS = 40
LR = 1e-3
TEMPERATURE = 6.0   # Softens teacher distribution (higher = softer)
ALPHA = 0.7         # Weight of KD loss vs hard-label loss

DATA_DIR = Path("data/processed")
TEACHER_PATH = Path("models/teacher/yamnet_finetuned.keras")
STUDENT_DIR = Path("models/student")
# ─────────────────────────────────────────────────────────────────────────────


class DistillationModel(tf.keras.Model):
    """Wraps student + teacher for joint training with KD loss."""

    def __init__(self, student, teacher, temperature, alpha):
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.temperature = temperature
        self.alpha = alpha

    def compile(self, optimizer, metrics):
        super().compile(optimizer=optimizer, metrics=metrics)
        self.kd_loss_fn = tf.keras.losses.KLDivergence()
        self.ce_loss_fn = tf.keras.losses.CategoricalCrossentropy()

    def train_step(self, data):
        x, y = data

        # Teacher soft labels (no gradient)
        teacher_logits = self.teacher(x, training=False)
        teacher_soft = tf.nn.softmax(teacher_logits / self.temperature)

        with tf.GradientTape() as tape:
            student_logits = self.student(x, training=True)
            student_soft = tf.nn.softmax(student_logits / self.temperature)

            # KD loss (soft labels)
            kd_loss = self.kd_loss_fn(teacher_soft, student_soft) * (self.temperature ** 2)

            # Hard label loss
            ce_loss = self.ce_loss_fn(y, student_logits)

            total_loss = self.alpha * kd_loss + (1 - self.alpha) * ce_loss

        grads = tape.gradient(total_loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.student.trainable_variables))

        self.compiled_metrics.update_state(y, student_logits)
        results = {m.name: m.result() for m in self.metrics}
        results["loss"] = total_loss
        results["kd_loss"] = kd_loss
        results["ce_loss"] = ce_loss
        return results

    def test_step(self, data):
        x, y = data
        student_logits = self.student(x, training=False)
        ce_loss = self.ce_loss_fn(y, student_logits)
        self.compiled_metrics.update_state(y, student_logits)
        results = {m.name: m.result() for m in self.metrics}
        results["loss"] = ce_loss
        return results


def train_with_distillation():
    STUDENT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    X_train = np.load(DATA_DIR / "X_train.npy")
    y_train = np.load(DATA_DIR / "y_train.npy")
    X_val = np.load(DATA_DIR / "X_val.npy")
    y_val = np.load(DATA_DIR / "y_val.npy")

    y_train_oh = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_val_oh = tf.keras.utils.to_categorical(y_val, NUM_CLASSES)

    input_shape = X_train.shape[1:]

    # Load teacher
    print("Loading teacher model...")
    teacher = tf.keras.models.load_model(str(TEACHER_PATH))
    teacher.trainable = False
    print(f"Teacher params: {teacher.count_params():,}")

    # Build student
    student = build_dscnn(input_shape, NUM_CLASSES)
    print(f"Student params: {student.count_params():,}")

    # Distillation model
    distill_model = DistillationModel(student, teacher, TEMPERATURE, ALPHA)
    distill_model.compile(
        optimizer=tf.keras.optimizers.Adam(LR),
        metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy")],
    )

    train_ds = (
        tf.data.Dataset.from_tensor_slices((X_train, y_train_oh))
        .shuffle(2000)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices((X_val, y_val_oh))
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=7, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(
            str(STUDENT_DIR / "best_student.keras"),
            save_best_only=True, monitor="val_accuracy"
        ),
    ]

    print(f"\n=== Knowledge Distillation (T={TEMPERATURE}, α={ALPHA}) ===")
    distill_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    student.save(STUDENT_DIR / "dscnn_student.keras")
    print(f"\nStudent saved → {STUDENT_DIR}/dscnn_student.keras")
    return student


if __name__ == "__main__":
    train_with_distillation()
