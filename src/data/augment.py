"""
Data augmentation pipeline.
Multiplies each raw clip into ~6 variants:
  1. Original (normalized)
  2. Pitch shift up (+2 semitones)
  3. Pitch shift down (−2 semitones)
  4. Time stretch (×0.85)
  5. Add white noise (SNR ~15dB)
  6. Add background noise (SNR ~10dB)

Run AFTER placing raw files in data/raw/<class>/
Output: data/raw/<class>/aug_*.wav (added alongside originals)

Usage: python src/data/augment.py
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
import random

SAMPLE_RATE = 16000
CLASSES = ["fire_alarm", "baby_cry", "choking", "car_horn", "doorbell", "background"]
RAW_DIR = Path("data/raw")
TARGET_PER_CLASS = 400  # Stop augmenting when we reach this


def load(path: str) -> np.ndarray:
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    # Normalize
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))
    return audio


def add_noise(audio: np.ndarray, snr_db: float = 15.0) -> np.ndarray:
    signal_power = np.mean(audio ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), len(audio))
    return np.clip(audio + noise, -1, 1)


def add_background(audio: np.ndarray, bg_dir: Path, snr_db: float = 10.0) -> np.ndarray:
    bg_files = list(bg_dir.glob("*.wav")) + list(bg_dir.glob("*.mp3"))
    if not bg_files:
        return add_noise(audio, snr_db)
    bg_path = random.choice(bg_files)
    bg, _ = librosa.load(str(bg_path), sr=SAMPLE_RATE, mono=True)
    # Loop background if shorter
    if len(bg) < len(audio):
        bg = np.tile(bg, int(np.ceil(len(audio) / len(bg))))
    bg = bg[:len(audio)]
    # Mix at target SNR
    sig_rms = np.sqrt(np.mean(audio ** 2)) + 1e-9
    bg_rms = np.sqrt(np.mean(bg ** 2)) + 1e-9
    target_bg_rms = sig_rms / (10 ** (snr_db / 20))
    bg = bg * (target_bg_rms / bg_rms)
    return np.clip(audio + bg, -1, 1)


def augment_clip(audio: np.ndarray, bg_dir: Path) -> list:
    """Returns list of (suffix, augmented_audio) pairs."""
    variants = []

    # Pitch shift
    try:
        variants.append(("ps_up", librosa.effects.pitch_shift(audio, sr=SAMPLE_RATE, n_steps=2)))
        variants.append(("ps_dn", librosa.effects.pitch_shift(audio, sr=SAMPLE_RATE, n_steps=-2)))
    except Exception:
        pass

    # Time stretch
    try:
        stretched = librosa.effects.time_stretch(audio, rate=0.85)
        if len(stretched) > len(audio):
            stretched = stretched[:len(audio)]
        else:
            stretched = np.pad(stretched, (0, len(audio) - len(stretched)))
        variants.append(("ts", stretched))
    except Exception:
        pass

    # Noise augmentation
    variants.append(("wn", add_noise(audio, snr_db=15.0)))
    variants.append(("bg", add_background(audio, bg_dir, snr_db=10.0)))

    return variants


def augment_class(cls: str):
    cls_dir = RAW_DIR / cls
    bg_dir = RAW_DIR / "background"

    existing = [f for f in cls_dir.glob("*.*")
                if f.suffix in {".wav", ".mp3", ".m4a", ".ogg", ".flac"}
                and not f.stem.startswith("aug_")]

    if not existing:
        print(f"  {cls:15s}: no source files, skipping")
        return

    # Count current total
    all_files = list(cls_dir.glob("*.*"))
    current = len([f for f in all_files if f.suffix in {".wav", ".mp3", ".m4a", ".ogg", ".flac"}])

    if current >= TARGET_PER_CLASS:
        print(f"  {cls:15s}: {current} files (target reached)")
        return

    needed = TARGET_PER_CLASS - current
    print(f"  {cls:15s}: {len(existing)} source → generating {needed} more")

    generated = 0
    for src in tqdm(existing, desc=f"    {cls}", leave=False):
        if generated >= needed:
            break
        try:
            audio = load(str(src))
            variants = augment_clip(audio, bg_dir)
            for suffix, aug_audio in variants:
                if generated >= needed:
                    break
                out = cls_dir / f"aug_{suffix}_{src.stem}.wav"
                if not out.exists():
                    sf.write(str(out), aug_audio, SAMPLE_RATE)
                    generated += 1
        except Exception as e:
            print(f"    Error {src.name}: {e}")

    final = len(list(cls_dir.glob("*.*")))
    print(f"  {cls:15s}: {final} files total")


def run():
    print("=== Data Augmentation ===")
    print(f"Target: {TARGET_PER_CLASS} files per class\n")
    for cls in CLASSES:
        augment_class(cls)
    print("\nDone. Final counts:")
    for cls in CLASSES:
        d = RAW_DIR / cls
        n = len(list(d.glob("*.*"))) if d.exists() else 0
        status = "OK" if n >= TARGET_PER_CLASS else f"LOW ({n})"
        print(f"  {cls:15s}: {n:4d}  [{status}]")


if __name__ == "__main__":
    run()
