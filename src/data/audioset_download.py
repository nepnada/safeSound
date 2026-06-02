"""
Downloads AudioSet clips for our 6 classes using yt-dlp.
AudioSet provides YouTube IDs + timestamps — we download and trim each clip.

Usage:
    python src/data/audioset_download.py
"""

import os
import csv
import subprocess
import urllib.request
from pathlib import Path
from tqdm import tqdm

RAW_DIR = Path("data/raw")
TMP_DIR = Path("data/tmp/audioset")

# AudioSet ontology IDs → our class names
# Source: https://research.google.com/audioset/ontology/
AUDIOSET_LABELS = {
    "/m/0k4j":   "fire_alarm",   # Fire alarm
    "/m/01y3hg": "fire_alarm",   # Smoke detector / alarm
    "/m/07qrkrw": "baby_cry",    # Baby cry / infant cry
    "/m/03qtwd":  "baby_cry",    # Baby laughter (extra)
    "/m/07plz5l": "choking",     # Cough
    "/m/01j3sz":  "choking",     # Gasp
    "/m/02mfyn":  "car_horn",    # Car horn
    "/m/03cl9h":  "car_horn",    # Vehicle horn, car horn, honking
    "/m/0fqfqc":  "doorbell",    # Doorbell
    "/m/07rjzl8": "background",  # Silence
    "/m/0l15bq":  "background",  # Humming
}

AUDIOSET_CSV_URL = (
    "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/balanced_train_segments.csv"
)


def download_csv():
    dest = TMP_DIR / "balanced_train_segments.csv"
    if dest.exists():
        return dest
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    print("Downloading AudioSet balanced train CSV...")
    urllib.request.urlretrieve(AUDIOSET_CSV_URL, dest)
    return dest


def parse_csv(csv_path: Path) -> dict:
    """Returns {class_name: [(ytid, start, end), ...]}"""
    clips = {c: [] for c in set(AUDIOSET_LABELS.values())}
    MAX_PER_CLASS = 300

    with open(csv_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            parts = line.split(", ")
            if len(parts) < 4:
                continue
            ytid, start, end = parts[0], float(parts[1]), float(parts[2])
            labels = parts[3].replace('"', "").split(",")
            for lbl in labels:
                lbl = lbl.strip()
                if lbl in AUDIOSET_LABELS:
                    cls = AUDIOSET_LABELS[lbl]
                    if len(clips[cls]) < MAX_PER_CLASS:
                        clips[cls].append((ytid, start, end))
    return clips


def download_clip(ytid: str, start: float, end: float, out_path: Path) -> bool:
    if out_path.exists():
        return True
    url = f"https://www.youtube.com/watch?v={ytid}"
    duration = end - start
    tmp = out_path.with_suffix(".tmp.%(ext)s")
    cmd = [
        "yt-dlp",
        "-x", "--audio-format", "wav",
        "--audio-quality", "0",
        "--postprocessor-args", f"ffmpeg:-ss {start} -t {duration} -ar 16000 -ac 1",
        "-o", str(tmp),
        "--quiet", "--no-warnings",
        url,
    ]
    try:
        result = subprocess.run(cmd, timeout=60, capture_output=True)
        # Find the downloaded file
        for f in out_path.parent.iterdir():
            if f.stem == out_path.stem + ".tmp" or "tmp" in f.stem:
                f.rename(out_path)
                return True
        # yt-dlp may name differently
        wav_tmp = out_path.with_name(out_path.stem + ".tmp.wav")
        if wav_tmp.exists():
            wav_tmp.rename(out_path)
            return True
        return result.returncode == 0
    except Exception:
        return False


def download_all():
    csv_path = download_csv()
    clips = parse_csv(csv_path)

    for cls, items in clips.items():
        out_dir = RAW_DIR / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{cls}: {len(items)} clips to download")
        success = 0
        for ytid, start, end in tqdm(items, desc=cls):
            fname = f"audioset_{ytid}_{int(start)}.wav"
            out_path = out_dir / fname
            if download_clip(ytid, start, end, out_path):
                success += 1
        print(f"  {success}/{len(items)} downloaded")


if __name__ == "__main__":
    download_all()
