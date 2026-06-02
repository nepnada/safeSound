"""
Dataset download helper.
Downloads ESC-50 and UrbanSound8K automatically.
For AudioSet: prints instructions (requires Google credentials).

Usage:
    python src/data/download.py --dataset esc50
    python src/data/download.py --dataset urbansound
    python src/data/download.py --all
"""

import os
import argparse
import zipfile
import tarfile
import shutil
import requests
from pathlib import Path
from tqdm import tqdm

RAW_DIR = Path("data/raw")

# ESC-50 class mappings relevant to our project
ESC50_CLASSES = {
    "fire_crackling": "fire_alarm",
    "crackling_fire": "fire_alarm",
    "baby_cry": "baby_cry",
    "crying_baby": "baby_cry",
    "car_horn": "car_horn",
    "door_knock": "doorbell",
    "doorbell": "doorbell",
}

# UrbanSound8K class mappings
US8K_CLASSES = {
    "car_horn": "car_horn",
}


def download_file(url: str, dest: Path, desc: str = ""):
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=desc) as bar:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))


def download_esc50():
    print("\n=== ESC-50 ===")
    url = "https://github.com/karoldvl/ESC-50/archive/master.zip"
    dest = Path("data/tmp/esc50.zip")

    if not dest.exists():
        download_file(url, dest, "ESC-50")

    print("Extracting...")
    with zipfile.ZipFile(dest) as z:
        z.extractall("data/tmp/esc50_raw")

    # Sort into our class folders
    audio_dir = Path("data/tmp/esc50_raw/ESC-50-master/audio")
    meta_file = Path("data/tmp/esc50_raw/ESC-50-master/meta/esc50.csv")

    import csv
    count = 0
    with open(meta_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            cat = row["category"].lower().replace(" ", "_")
            if cat in ESC50_CLASSES:
                target_class = ESC50_CLASSES[cat]
                src = audio_dir / row["filename"]
                dst_dir = RAW_DIR / target_class
                dst_dir.mkdir(parents=True, exist_ok=True)
                dst = dst_dir / f"esc50_{row['filename']}"
                if src.exists() and not dst.exists():
                    shutil.copy(src, dst)
                    count += 1

    print(f"  Copied {count} files from ESC-50")


def download_urbansound8k():
    print("\n=== UrbanSound8K ===")
    print("  UrbanSound8K requires manual download from:")
    print("  https://urbansounddataset.weebly.com/urbansound8k.html")
    print("  After downloading, extract and place .wav files in:")
    print("  data/raw/car_horn/")
    print("  (UrbanSound8K class 3 = car_horn)")


def download_demand():
    print("\n=== DEMAND (background noise) ===")
    print("  DEMAND dataset: https://zenodo.org/record/1227121")
    print("  Download any environments (CAFE, HOME, STREET, etc.)")
    print("  Place .wav files in: data/raw/background/")


def print_audioset_instructions():
    print("\n=== AudioSet (fire alarm, choking, baby cry) ===")
    print("  AudioSet requires downloading via YouTube IDs.")
    print("  Use: https://github.com/marl/audiosetdl")
    print("  Or manually search and download from Freesound:")
    print("    - fire alarm: https://freesound.org/search/?q=fire+alarm")
    print("    - choking:    https://freesound.org/search/?q=choking+gagging")
    print("    - baby crying: https://freesound.org/search/?q=baby+crying")
    print("  Place files in data/raw/<class>/")


def check_status():
    print("\n=== Dataset Status ===")
    for cls in ["fire_alarm", "baby_cry", "choking", "car_horn", "doorbell", "background"]:
        d = RAW_DIR / cls
        if d.exists():
            files = [f for f in d.iterdir() if f.suffix in {".wav", ".mp3", ".ogg", ".flac"}]
            print(f"  {cls:15s}: {len(files):4d} files")
        else:
            print(f"  {cls:15s}: missing folder")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["esc50", "urbansound", "all"])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    Path("data/tmp").mkdir(parents=True, exist_ok=True)

    if args.status:
        check_status()
    elif args.dataset == "esc50" or args.all:
        download_esc50()
        if args.all:
            download_urbansound8k()
            download_demand()
            print_audioset_instructions()
        check_status()
    elif args.dataset == "urbansound":
        download_urbansound8k()
    else:
        print_audioset_instructions()
        download_urbansound8k()
        download_demand()
        check_status()
