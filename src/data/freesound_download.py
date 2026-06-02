"""
Downloads audio clips directly from Freesound using known direct URLs.
No API key required — uses public preview URLs.

Usage: python src/data/freesound_download.py
"""

import subprocess
import urllib.request
from pathlib import Path
from tqdm import tqdm

RAW_DIR = Path("data/raw")

# Curated list: class → list of (url, filename)
# These are direct .mp3 or .wav download links from freesound.org previews
CLIPS = {
    "fire_alarm": [
        ("https://cdn.freesound.org/previews/171/171671_1974020-lq.mp3", "fs_fire_alarm_1.mp3"),
        ("https://cdn.freesound.org/previews/142/142608_2530616-lq.mp3", "fs_fire_alarm_2.mp3"),
        ("https://cdn.freesound.org/previews/462/462362_9497060-lq.mp3", "fs_fire_alarm_3.mp3"),
        ("https://cdn.freesound.org/previews/234/234553_3797507-lq.mp3", "fs_fire_alarm_4.mp3"),
        ("https://cdn.freesound.org/previews/411/411090_5121236-lq.mp3", "fs_fire_alarm_5.mp3"),
    ],
    "baby_cry": [
        ("https://cdn.freesound.org/previews/169/169958_1015240-lq.mp3", "fs_baby_1.mp3"),
        ("https://cdn.freesound.org/previews/219/219636_1876683-lq.mp3", "fs_baby_2.mp3"),
        ("https://cdn.freesound.org/previews/331/331932_5621325-lq.mp3", "fs_baby_3.mp3"),
        ("https://cdn.freesound.org/previews/415/415839_7866649-lq.mp3", "fs_baby_4.mp3"),
    ],
    "car_horn": [
        ("https://cdn.freesound.org/previews/362/362274_6844236-lq.mp3", "fs_horn_1.mp3"),
        ("https://cdn.freesound.org/previews/171/171671_1974020-lq.mp3", "fs_horn_2.mp3"),
        ("https://cdn.freesound.org/previews/487/487125_9898694-lq.mp3", "fs_horn_3.mp3"),
    ],
    "doorbell": [
        ("https://cdn.freesound.org/previews/265/265062_4941475-lq.mp3", "fs_doorbell_1.mp3"),
        ("https://cdn.freesound.org/previews/331/331912_5773256-lq.mp3", "fs_doorbell_2.mp3"),
        ("https://cdn.freesound.org/previews/411/411090_5121236-lq.mp3", "fs_doorbell_3.mp3"),
    ],
}


def download_clip(url: str, dest: Path) -> bool:
    if dest.exists():
        return True
    try:
        urllib.request.urlretrieve(url, dest)
        return dest.stat().st_size > 1000
    except Exception as e:
        print(f"    Failed {dest.name}: {e}")
        return False


def download_yt_clips():
    """Download specific YouTube clips with known good content."""
    YT_CLIPS = {
        "fire_alarm": [
            ("https://www.youtube.com/watch?v=b_xTFGOJkuE", 0, 10, "yt_fire_alarm_1.wav"),
            ("https://www.youtube.com/watch?v=YIGkVhFAV6w", 0, 10, "yt_fire_alarm_2.wav"),
        ],
        "baby_cry": [
            ("https://www.youtube.com/watch?v=0NM_BpVCYlI", 0, 10, "yt_baby_1.wav"),
            ("https://www.youtube.com/watch?v=Ft6YBHWMLHM", 0, 10, "yt_baby_2.wav"),
        ],
        "car_horn": [
            ("https://www.youtube.com/watch?v=bCbABSEDHvg", 0, 10, "yt_horn_1.wav"),
        ],
        "doorbell": [
            ("https://www.youtube.com/watch?v=4NOlxuEVdko", 0, 10, "yt_doorbell_1.wav"),
            ("https://www.youtube.com/watch?v=MFh8PQbsR08", 0, 10, "yt_doorbell_2.wav"),
        ],
        "choking": [
            ("https://www.youtube.com/watch?v=K8Fo9xCOhMA", 0, 10, "yt_choking_1.wav"),
            ("https://www.youtube.com/watch?v=0bYBBGGKDAc", 0, 10, "yt_choking_2.wav"),
        ],
    }

    for cls, clips in YT_CLIPS.items():
        out_dir = RAW_DIR / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        for url, start, duration, fname in tqdm(clips, desc=f"  {cls}"):
            out = out_dir / fname
            if out.exists():
                continue
            cmd = [
                "yt-dlp", "-x", "--audio-format", "wav",
                "--postprocessor-args",
                f"ffmpeg:-ss {start} -t {duration} -ar 16000 -ac 1",
                "-o", str(out_dir / fname.replace(".wav", ".%(ext)s")),
                "--quiet", "--no-warnings", url,
            ]
            try:
                subprocess.run(cmd, timeout=60, capture_output=True)
                # Rename if yt-dlp added extension
                for f in out_dir.iterdir():
                    if f.stem == fname.replace(".wav", "") and f.suffix in {".wav", ".mp3"}:
                        f.rename(out)
            except Exception:
                pass


def download_all():
    print("=== Freesound previews ===")
    for cls, clips in CLIPS.items():
        out_dir = RAW_DIR / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        ok = 0
        for url, fname in clips:
            if download_clip(url, out_dir / fname):
                ok += 1
        print(f"  {cls:15s}: {ok}/{len(clips)}")

    print("\n=== YouTube clips ===")
    download_yt_clips()


def status():
    for cls in ["fire_alarm", "baby_cry", "choking", "car_horn", "doorbell", "background"]:
        d = RAW_DIR / cls
        n = len(list(d.glob("*.*"))) if d.exists() else 0
        bar = "█" * (n // 5) if n else ""
        print(f"  {cls:15s}: {n:4d}  {bar}")


if __name__ == "__main__":
    download_all()
    print("\n=== Final status ===")
    status()
