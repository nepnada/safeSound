"""
Audio preprocessing pipeline.
Input: raw .wav/.mp3 files in data/raw/<class>/
Output: numpy arrays in data/processed/
"""

import os
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from tqdm import tqdm

# ── Config ──────────────────────────────────────────────────────────────────
SAMPLE_RATE = 16000
WINDOW_SEC = 1.0
HOP_SEC = 0.5          # 50% overlap for augmentation
N_MELS = 64
N_FFT = 512            # 32ms at 16kHz
HOP_LENGTH = 256       # 16ms at 16kHz
F_MIN = 60.0
F_MAX = 7600.0

CLASSES = ["fire_alarm", "baby_cry", "choking", "car_horn", "doorbell", "background"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
# ─────────────────────────────────────────────────────────────────────────────


def load_and_resample(path: str) -> np.ndarray:
    audio, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    return audio


def audio_to_logmel(audio: np.ndarray) -> np.ndarray:
    """Convert 1s audio clip → log-mel spectrogram (64 x T)."""
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=F_MIN,
        fmax=F_MAX,
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    # Normalize to [0, 1]
    log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-8)
    return log_mel.astype(np.float32)


def segment_audio(audio: np.ndarray) -> list:
    """Slice audio into overlapping 1s windows."""
    window = int(WINDOW_SEC * SAMPLE_RATE)
    hop = int(HOP_SEC * SAMPLE_RATE)
    segments = []
    for start in range(0, len(audio) - window + 1, hop):
        seg = audio[start : start + window]
        segments.append(seg)
    # Keep last segment even if short (zero-pad)
    if len(audio) >= window // 2:
        last = audio[-window:]
        if len(last) < window:
            last = np.pad(last, (0, window - len(last)))
        segments.append(last)
    return segments


def process_class(class_name: str) -> tuple:
    class_dir = RAW_DIR / class_name
    if not class_dir.exists():
        print(f"  [SKIP] {class_name}: folder not found")
        return np.array([]), np.array([])

    files = [f for f in class_dir.iterdir() if f.suffix in {".wav", ".mp3", ".ogg", ".flac", ".m4a"}]
    if not files:
        print(f"  [SKIP] {class_name}: no audio files")
        return np.array([]), np.array([])

    # Separate source files from augmented to avoid data leakage
    # Source files go to val/test, augmented files only in train
    source_files = [f for f in files if not f.stem.startswith("aug_")]
    aug_files    = [f for f in files if f.stem.startswith("aug_")]

    label = CLASS_TO_IDX[class_name]
    source_feats, aug_feats, labels_src, labels_aug = [], [], [], []

    for f in tqdm(source_files, desc=f"  {class_name} (src)", leave=False):
        try:
            audio = load_and_resample(str(f))
            for seg in segment_audio(audio):
                source_feats.append(audio_to_logmel(seg))
                labels_src.append(label)
        except Exception as e:
            print(f"    [ERROR] {f.name}: {e}")

    for f in tqdm(aug_files, desc=f"  {class_name} (aug)", leave=False):
        try:
            audio = load_and_resample(str(f))
            for seg in segment_audio(audio):
                aug_feats.append(audio_to_logmel(seg))
                labels_aug.append(label)
        except Exception as e:
            pass  # silently skip bad aug files

    src = (np.array(source_feats), np.array(labels_src))
    aug = (np.array(aug_feats),    np.array(labels_aug))
    return src, aug


def build_dataset(test_split: float = 0.15, val_split: float = 0.15):
    """
    Split strategy to avoid data leakage:
    - Source files → split into val + test + part of train
    - Augmented files → train only (never in val/test)
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    src_features, src_labels = [], []
    aug_features, aug_labels = [], []

    print("Processing classes...")
    for cls in CLASSES:
        result = process_class(cls)
        if result is None:
            continue
        (src_X, src_y), (aug_X, aug_y) = result
        if len(src_X):
            src_features.append(src_X)
            src_labels.append(src_y)
        if len(aug_X):
            aug_features.append(aug_X)
            aug_labels.append(aug_y)
        print(f"  {cls}: {len(src_X)} source segs + {len(aug_X)} aug segs")

    Xs = np.concatenate(src_features, axis=0)[..., np.newaxis]
    ys = np.concatenate(src_labels, axis=0)
    Xa = np.concatenate(aug_features, axis=0)[..., np.newaxis] if aug_features else np.empty((0,))
    ya = np.concatenate(aug_labels, axis=0) if aug_labels else np.empty((0,))

    print(f"\nSource: {len(Xs)} | Augmented: {len(Xa)}")

    # Shuffle source, then split
    idx = np.random.permutation(len(Xs))
    Xs, ys = Xs[idx], ys[idx]

    n = len(Xs)
    n_test = int(n * test_split)
    n_val  = int(n * val_split)

    X_test,  y_test  = Xs[:n_test],           ys[:n_test]
    X_val,   y_val   = Xs[n_test:n_test+n_val], ys[n_test:n_test+n_val]
    X_src_train, y_src_train = Xs[n_test+n_val:], ys[n_test+n_val:]

    # Train = source remainder + ALL augmented
    if len(Xa):
        idx_a = np.random.permutation(len(Xa))
        Xa, ya = Xa[idx_a], ya[idx_a]
        X_train = np.concatenate([X_src_train, Xa], axis=0)
        y_train = np.concatenate([y_src_train, ya], axis=0)
    else:
        X_train, y_train = X_src_train, y_src_train

    # Final shuffle of train
    idx_t = np.random.permutation(len(X_train))
    X_train, y_train = X_train[idx_t], y_train[idx_t]

    for name, arr in [("X_train", X_train), ("y_train", y_train),
                      ("X_val",   X_val),   ("y_val",   y_val),
                      ("X_test",  X_test),  ("y_test",  y_test)]:
        np.save(OUT_DIR / f"{name}.npy", arr)

    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    print(f"Saved to {OUT_DIR}/")


if __name__ == "__main__":
    build_dataset()
