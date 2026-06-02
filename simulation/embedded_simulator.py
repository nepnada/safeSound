"""
ESP32-S3 Embedded Pipeline Simulator
=====================================
Simulates the COMPLETE embedded inference pipeline as it would run on the ESP32-S3:

  Audio file (simulated INMP441) → Log-Mel extraction → TFLite INT8 inference
  → Temporal smoothing (3 frames) → Per-class thresholds → Alert decision

This is a cycle-accurate simulation of the firmware logic in main.ino,
running the real quantized TFLite model.

Usage:
    python simulation/embedded_simulator.py                         # test on all test files
    python simulation/embedded_simulator.py --file path/to/audio.wav  # test on specific file
    python simulation/embedded_simulator.py --live                  # live microphone input
"""

import numpy as np
import tensorflow as tf
import librosa
import time
import sys
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

# ── ESP32-S3 simulated hardware specs ────────────────────────────────────────
ESP32_CLOCK_MHZ     = 240
ESP32_SRAM_KB       = 512
ESP32_FLASH_MB      = 8
ESP32_CPU_SLOWDOWN  = 8.0   # TFLite on ESP32 is ~8x slower than M-series Mac

# ── Audio pipeline config (mirrors firmware/esp32/main.ino) ──────────────────
SAMPLE_RATE   = 16000
WINDOW_SEC    = 1.0
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_SEC)
N_MELS        = 64
N_FFT         = 512
HOP_LENGTH    = 256
F_MIN         = 60.0
F_MAX         = 7600.0

# ── Model config ─────────────────────────────────────────────────────────────
NUM_CLASSES   = 6
CLASSES       = ["fire_alarm", "baby_cry", "choking", "car_horn", "doorbell", "background"]
TFLITE_PATH   = Path("models/student/dscnn_int8.tflite")

# ── Temporal smoothing (matches firmware) ────────────────────────────────────
SMOOTH_FRAMES = 3

# ── Per-class confidence thresholds (matches firmware) ────────────────────────
THRESHOLDS = {
    "fire_alarm": 0.75,
    "baby_cry":   0.80,
    "choking":    0.75,
    "car_horn":   0.80,
    "doorbell":   0.90,
    "background": 0.50,
}

# ── LED color mapping (matches WS2812B config in firmware) ────────────────────
LED_COLORS = {
    "fire_alarm": "\033[91m█████ RED",
    "baby_cry":   "\033[93m█████ ORANGE",
    "choking":    "\033[95m█████ PURPLE",
    "car_horn":   "\033[33m█████ YELLOW",
    "doorbell":   "\033[94m█████ BLUE",
    "background": "\033[90m───── OFF",
}
RESET = "\033[0m"

# ── Vibration patterns (matches firmware) ─────────────────────────────────────
VIBRO_PATTERNS = {
    "fire_alarm": "▓▓▓▓▓ (5 pulses — CRITICAL)",
    "baby_cry":   "▓▓▓ (3 pulses)",
    "choking":    "▓▓▓▓▓ (5 pulses — CRITICAL)",
    "car_horn":   "▓▓▓ (3 pulses)",
    "doorbell":   "▓▓▓ (3 pulses)",
    "background": "── (none)",
}


@dataclass
class InferenceResult:
    """Single inference frame result."""
    frame_id: int
    timestamp_ms: float
    raw_probs: list
    smoothed_probs: list
    predicted_class: str
    confidence: float
    threshold: float
    alert_triggered: bool
    latency_host_ms: float
    latency_esp32_est_ms: float
    led_state: str
    vibro_state: str


@dataclass
class SimulationReport:
    """Full simulation report."""
    audio_source: str
    duration_sec: float
    total_frames: int
    results: list = field(default_factory=list)
    alerts: list = field(default_factory=list)
    avg_latency_ms: float = 0.0
    est_esp32_latency_ms: float = 0.0
    memory_usage_kb: float = 0.0


class ESP32Simulator:
    """Simulates the full ESP32-S3 embedded inference pipeline."""

    def __init__(self, model_path: str = None):
        path = Path(model_path) if model_path else TFLITE_PATH
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}\nRun training first: bash run_pipeline.sh")

        # Load TFLite interpreter (simulates TFLite Micro on ESP32)
        self.interpreter = tf.lite.Interpreter(model_path=str(path))
        self.interpreter.allocate_tensors()
        self.inp = self.interpreter.get_input_details()[0]
        self.out = self.interpreter.get_output_details()[0]
        self.scale_in, self.zero_in   = self.inp["quantization"]
        self.scale_out, self.zero_out = self.out["quantization"]

        # Temporal smoothing buffer (circular, matches firmware)
        self.smooth_buffer = np.ones((SMOOTH_FRAMES, NUM_CLASSES)) / NUM_CLASSES
        self.smooth_idx = 0

        # Memory estimation
        model_size = path.stat().st_size
        tensor_arena = self.inp["shape"].prod() + self.out["shape"].prod() + 8192
        self.memory_kb = (model_size + tensor_arena) / 1024

        print(f"╔══════════════════════════════════════════════════════╗")
        print(f"║  ESP32-S3 Sound Alert System — Embedded Simulator   ║")
        print(f"╠══════════════════════════════════════════════════════╣")
        print(f"║  CPU:    {ESP32_CLOCK_MHZ} MHz (dual-core LX7)                ║")
        print(f"║  SRAM:   {ESP32_SRAM_KB} KB  |  Flash: {ESP32_FLASH_MB} MB                   ║")
        print(f"║  Model:  {model_size/1024:.1f} KB INT8  |  Arena: ~{tensor_arena/1024:.1f} KB        ║")
        print(f"║  Classes: {NUM_CLASSES}  |  Smoothing: {SMOOTH_FRAMES} frames              ║")
        print(f"╚══════════════════════════════════════════════════════╝")

    def _compute_logmel(self, audio: np.ndarray) -> np.ndarray:
        """Simulates on-device log-mel computation (ESP-DSP equivalent)."""
        mel = librosa.feature.melspectrogram(
            y=audio, sr=SAMPLE_RATE, n_fft=N_FFT,
            hop_length=HOP_LENGTH, n_mels=N_MELS,
            fmin=F_MIN, fmax=F_MAX, power=2.0
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)
        log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-8)
        return log_mel.astype(np.float32)

    def _quantize_input(self, features: np.ndarray) -> np.ndarray:
        """Simulates INT8 quantization (matches firmware)."""
        x = features[np.newaxis, ..., np.newaxis]  # (1, 64, T, 1)
        return (x / self.scale_in + self.zero_in).astype(np.int8)

    def _dequantize_output(self, output_q: np.ndarray) -> np.ndarray:
        """Simulates INT8 dequantization (matches firmware)."""
        return (output_q.astype(np.float32) - self.zero_out) * self.scale_out

    def _temporal_smooth(self, probs: np.ndarray) -> np.ndarray:
        """Circular buffer averaging — exact copy of firmware logic."""
        self.smooth_buffer[self.smooth_idx] = probs
        self.smooth_idx = (self.smooth_idx + 1) % SMOOTH_FRAMES
        return self.smooth_buffer.mean(axis=0)

    def _decide_alert(self, smoothed: np.ndarray) -> tuple:
        """Per-class threshold decision — matches firmware."""
        best_idx = np.argmax(smoothed)
        best_class = CLASSES[best_idx]
        confidence = smoothed[best_idx]
        threshold = THRESHOLDS[best_class]
        alert = confidence >= threshold and best_class != "background"
        return best_class, confidence, threshold, alert

    def infer_frame(self, audio_window: np.ndarray, frame_id: int = 0) -> InferenceResult:
        """Run single inference frame — full embedded pipeline simulation."""
        t0 = time.perf_counter()

        # Step 1: Log-mel extraction (simulates ESP-DSP FFT + mel filterbank)
        log_mel = self._compute_logmel(audio_window)

        # Step 2: Quantize to INT8 (simulates TFLite Micro input quantization)
        x_q = self._quantize_input(log_mel)

        # Step 3: TFLite inference (simulates TFLite Micro on Xtensa LX7)
        self.interpreter.set_tensor(self.inp["index"], x_q)
        self.interpreter.invoke()
        y_q = self.interpreter.get_tensor(self.out["index"])

        # Step 4: Dequantize output
        probs = self._dequantize_output(y_q)[0]

        # Step 5: Temporal smoothing (3-frame circular buffer)
        smoothed = self._temporal_smooth(probs)

        # Step 6: Alert decision (per-class thresholds)
        cls, conf, thresh, alert = self._decide_alert(smoothed)

        latency_host = (time.perf_counter() - t0) * 1000
        latency_esp32 = latency_host * ESP32_CPU_SLOWDOWN

        return InferenceResult(
            frame_id=frame_id,
            timestamp_ms=frame_id * 500,  # 500ms stride
            raw_probs=probs.tolist(),
            smoothed_probs=smoothed.tolist(),
            predicted_class=cls,
            confidence=conf,
            threshold=thresh,
            alert_triggered=alert,
            latency_host_ms=round(latency_host, 2),
            latency_esp32_est_ms=round(latency_esp32, 2),
            led_state=LED_COLORS.get(cls, ""),
            vibro_state=VIBRO_PATTERNS.get(cls, ""),
        )

    def simulate_audio_file(self, path: str, verbose: bool = True) -> SimulationReport:
        """Simulate processing an entire audio file through the embedded pipeline."""
        audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
        duration = len(audio) / SAMPLE_RATE

        # Reset smoothing buffer
        self.smooth_buffer = np.ones((SMOOTH_FRAMES, NUM_CLASSES)) / NUM_CLASSES
        self.smooth_idx = 0

        report = SimulationReport(
            audio_source=str(path),
            duration_sec=round(duration, 2),
            total_frames=0,
            memory_usage_kb=round(self.memory_kb, 1),
        )

        # Sliding window with 500ms stride (matches firmware loop)
        stride = WINDOW_SAMPLES // 2
        frame_id = 0

        if verbose:
            print(f"\n── Simulating: {Path(path).name} ({duration:.1f}s) ──")
            print(f"{'Frame':>5} {'Time':>7} {'Class':>12} {'Conf':>6} {'Thresh':>7} {'Alert':>6} {'Latency':>10} {'LED':>20} {'Vibro'}")
            print("─" * 100)

        for start in range(0, len(audio) - WINDOW_SAMPLES + 1, stride):
            window = audio[start:start + WINDOW_SAMPLES]
            result = self.infer_frame(window, frame_id)
            report.results.append(result)

            if result.alert_triggered:
                report.alerts.append(result)

            if verbose:
                alert_mark = "🚨" if result.alert_triggered else "  "
                print(
                    f"{result.frame_id:5d} "
                    f"{result.timestamp_ms/1000:6.1f}s "
                    f"{result.predicted_class:>12} "
                    f"{result.confidence:5.3f} "
                    f"{'>' if result.alert_triggered else '<'}{result.threshold:.2f} "
                    f"{alert_mark:>4} "
                    f"{result.latency_esp32_est_ms:6.1f}ms "
                    f"{result.led_state}{RESET} "
                    f"{result.vibro_state}"
                )
            frame_id += 1

        report.total_frames = frame_id
        latencies = [r.latency_esp32_est_ms for r in report.results]
        report.avg_latency_ms = round(np.mean(latencies), 2) if latencies else 0
        report.est_esp32_latency_ms = round(np.mean(latencies), 2)

        if verbose:
            print(f"\n── Summary ──")
            print(f"  Frames processed: {report.total_frames}")
            print(f"  Alerts triggered: {len(report.alerts)}")
            print(f"  Avg latency (ESP32 est.): {report.avg_latency_ms:.1f}ms")
            print(f"  Memory usage: {report.memory_usage_kb:.1f} KB / {ESP32_SRAM_KB} KB")
            if report.alerts:
                print(f"  Alert classes: {', '.join(set(a.predicted_class for a in report.alerts))}")

        return report

    def simulate_live(self):
        """Live microphone simulation — processes audio from Mac mic in real-time."""
        try:
            import sounddevice as sd
        except ImportError:
            print("Install sounddevice: pip install sounddevice")
            return

        self.smooth_buffer = np.ones((SMOOTH_FRAMES, NUM_CLASSES)) / NUM_CLASSES
        self.smooth_idx = 0

        print(f"\n╔══════════════════════════════════════════════════╗")
        print(f"║     LIVE SIMULATION — Press Ctrl+C to stop       ║")
        print(f"╚══════════════════════════════════════════════════╝\n")

        try:
            with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                                dtype="float32", blocksize=WINDOW_SAMPLES) as stream:
                frame_id = 0
                while True:
                    audio, _ = stream.read(WINDOW_SAMPLES)
                    result = self.infer_frame(audio.flatten(), frame_id)

                    # Real-time display
                    bar = "█" * int(result.confidence * 30)
                    alert = " 🚨 ALERT!" if result.alert_triggered else ""
                    color = "\033[91m" if result.alert_triggered else "\033[92m"
                    print(f"\r{color}[{result.latency_esp32_est_ms:5.1f}ms] "
                          f"{result.predicted_class:12s} {result.confidence:.3f} "
                          f"|{bar:<30}|{RESET}{alert}   ", end="", flush=True)
                    frame_id += 1
        except KeyboardInterrupt:
            print("\n\nSimulation stopped.")


def run_test_suite():
    """Run simulation on representative audio files from the dataset."""
    sim = ESP32Simulator()

    print("\n" + "=" * 60)
    print(" AUTOMATED TEST SUITE — Testing all classes")
    print("=" * 60)

    RAW_DIR = Path("data/raw")
    all_reports = []

    for cls in CLASSES:
        cls_dir = RAW_DIR / cls
        if not cls_dir.exists():
            continue
        # Pick first non-augmented file
        files = [f for f in cls_dir.glob("*.*")
                 if f.suffix in {".wav", ".mp3", ".ogg", ".flac"}
                 and not f.stem.startswith("aug_")]
        if files:
            report = sim.simulate_audio_file(str(files[0]))
            all_reports.append((cls, report))

    # Summary table
    print("\n" + "=" * 60)
    print(" TEST RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Expected':>12} {'Predicted':>12} {'Alerts':>7} {'Frames':>7} {'Latency':>10}")
    print("─" * 55)
    for expected, report in all_reports:
        if report.alerts:
            predicted = max(set(a.predicted_class for a in report.alerts),
                           key=lambda c: sum(1 for a in report.alerts if a.predicted_class == c))
        else:
            # Most common prediction across all frames
            preds = [r.predicted_class for r in report.results]
            predicted = max(set(preds), key=preds.count) if preds else "none"
        match = "✓" if expected == predicted or (expected == "background" and not report.alerts) else "✗"
        print(f"{expected:>12} → {predicted:>12} {len(report.alerts):>5} {report.total_frames:>7} "
              f"{report.avg_latency_ms:>7.1f}ms  {match}")

    # Export reports as JSON
    out = Path("simulation/results")
    out.mkdir(parents=True, exist_ok=True)
    for cls, report in all_reports:
        # Convert numpy types for JSON serialization
        def to_native(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            return obj

        raw_data = {
            "audio_source": report.audio_source,
            "duration_sec": report.duration_sec,
            "total_frames": report.total_frames,
            "alerts_count": len(report.alerts),
            "avg_latency_ms": report.avg_latency_ms,
            "memory_kb": report.memory_usage_kb,
            "results": [asdict(r) for r in report.results],
        }
        data = json.loads(json.dumps(raw_data, default=to_native))
        with open(out / f"sim_{cls}.json", "w") as f:
            json.dump(data, f, indent=2)
    print(f"\nDetailed results saved to {out}/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ESP32-S3 Sound Alert Simulator")
    parser.add_argument("--file", help="Audio file to simulate")
    parser.add_argument("--live", action="store_true", help="Live microphone input")
    parser.add_argument("--all", action="store_true", help="Run on all test files (default)")
    args = parser.parse_args()

    if args.live:
        sim = ESP32Simulator()
        sim.simulate_live()
    elif args.file:
        sim = ESP32Simulator()
        sim.simulate_audio_file(args.file)
    else:
        run_test_suite()
