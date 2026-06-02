"""
SafeSound — Live Sound Alert Dashboard (Flask + WebSocket)
===========================================================
Premium web dashboard with live microphone classification.
Replaces the Streamlit dashboard with a standalone app.

Run:
    python simulation/app.py

Then open http://localhost:5050 in your browser.
"""

import numpy as np
import tensorflow as tf
import librosa
import time
import json
import struct
import base64
import os
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from flask_sock import Sock

# ── Configuration (mirrors firmware/esp32/main.ino) ──────────────────────────
SAMPLE_RATE = 16000
WINDOW_SAMPLES = 16000  # 1 second
N_MELS = 64
N_FFT = 512
HOP_LENGTH = 256
F_MIN = 60.0
F_MAX = 7600.0

NUM_CLASSES = 6
CLASSES = ["fire_alarm", "baby_cry", "choking", "car_horn", "doorbell", "background"]
TFLITE_PATH = Path("models/student/dscnn_int8.tflite")

SMOOTH_FRAMES = 3
ESP32_CPU_SLOWDOWN = 8.0

# ESP32-S3 hardware specs for resource dashboard
ESP32_SRAM_KB = 512
ESP32_FLASH_MB = 8
ESP32_CLOCK_MHZ = 240
BATTERY_MAH = 1000
BATTERY_RUNTIME_H = 10  # estimated

THRESHOLDS = {
    "fire_alarm": 0.75,
    "baby_cry":   0.80,
    "choking":    0.75,
    "car_horn":   0.80,
    "doorbell":   0.90,
    "background": 0.50,
}

PRIORITY = {
    "fire_alarm": "CRITICAL",
    "choking":    "CRITICAL",
    "baby_cry":   "HIGH",
    "car_horn":   "HIGH",
    "doorbell":   "MEDIUM",
    "background": "NONE",
}

LED_COLORS = {
    "fire_alarm": "#D94F3B",
    "baby_cry":   "#E8913A",
    "choking":    "#D94F3B",
    "car_horn":   "#D4A843",
    "doorbell":   "#5B8FA8",
    "background": "#B0A89A",
}

VIBRATION_PATTERN = {
    "fire_alarm": "5 rapid pulses",
    "choking":    "5 rapid pulses",
    "baby_cry":   "3 pulses",
    "car_horn":   "3 pulses",
    "doorbell":   "1 long pulse",
    "background": "None",
}


# ── Model loading ────────────────────────────────────────────────────────────

def load_model():
    if not TFLITE_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {TFLITE_PATH}\n"
            "Run training first: bash run_pipeline.sh"
        )
    interp = tf.lite.Interpreter(model_path=str(TFLITE_PATH))
    interp.allocate_tensors()
    return interp


def compute_logmel(audio):
    mel = librosa.feature.melspectrogram(
        y=audio.astype(np.float32), sr=SAMPLE_RATE, n_fft=N_FFT,
        hop_length=HOP_LENGTH, n_mels=N_MELS, fmin=F_MIN, fmax=F_MAX, power=2.0
    )
    lm = librosa.power_to_db(mel, ref=np.max)
    lm = (lm - lm.min()) / (lm.max() - lm.min() + 1e-8)
    return lm.astype(np.float32)


def run_full_inference(audio, interp, inp_details, out_details,
                       scale_in, zero_in, scale_out, zero_out,
                       smooth_buffer, smooth_idx):
    """Run full inference pipeline on 1-second audio. Returns result dict + updated smooth_idx."""
    t0 = time.perf_counter()

    # 1. Log-mel spectrogram
    log_mel = compute_logmel(audio)

    # 2. Quantize input
    x = log_mel[np.newaxis, ..., np.newaxis]
    x_q = (x / scale_in + zero_in).astype(np.int8)

    # 3. TFLite inference
    interp.set_tensor(inp_details["index"], x_q)
    interp.invoke()
    y_q = interp.get_tensor(out_details["index"])

    # 4. Dequantize output
    probs = (y_q.astype(np.float32) - zero_out) * scale_out
    probs = probs[0]

    # 5. Temporal smoothing (3-frame circular buffer)
    smooth_buffer[smooth_idx] = probs
    smooth_idx = (smooth_idx + 1) % SMOOTH_FRAMES
    avg = smooth_buffer.mean(axis=0)

    # 6. Alert decision
    best_idx = int(np.argmax(avg))
    best_class = CLASSES[best_idx]
    confidence = float(avg[best_idx])
    threshold = THRESHOLDS[best_class]
    alert = confidence >= threshold and best_class != "background"

    latency_host = (time.perf_counter() - t0) * 1000
    latency_esp32 = latency_host * ESP32_CPU_SLOWDOWN

    # Serialize spectrogram as base64 for the frontend heatmap
    # Downsample to save bandwidth: take every other mel bin & time step
    spec_small = log_mel[::2, ::2].tolist()

    result = {
        "type": "result",
        "predicted_class": best_class,
        "confidence": round(confidence, 4),
        "threshold": threshold,
        "alert": alert,
        "priority": PRIORITY[best_class],
        "led_color": LED_COLORS[best_class],
        "vibration": VIBRATION_PATTERN[best_class],
        "probs": {c: round(float(avg[i]), 4) for i, c in enumerate(CLASSES)},
        "raw_probs": {c: round(float(probs[i]), 4) for i, c in enumerate(CLASSES)},
        "latency_ms": round(latency_host, 2),
        "est_esp32_ms": round(latency_esp32, 2),
        "timestamp": time.time(),
        "spectrogram": spec_small,
    }
    return result, smooth_idx


# ── Flask App ────────────────────────────────────────────────────────────────

app = Flask(__name__)
sock = Sock(app)

# Load model at startup
interpreter = load_model()
inp_details = interpreter.get_input_details()[0]
out_details = interpreter.get_output_details()[0]
scale_in, zero_in = inp_details["quantization"]
scale_out, zero_out = out_details["quantization"]

model_size_kb = TFLITE_PATH.stat().st_size / 1024
tensor_arena_kb = (inp_details["shape"].prod() + out_details["shape"].prod() + 8192) / 1024
total_mem_kb = model_size_kb + tensor_arena_kb

print(f"✓ Model loaded: {TFLITE_PATH} ({model_size_kb:.1f} KB)")
print(f"✓ Classes: {CLASSES}")
print(f"✓ Input shape: {inp_details['shape']}")
print(f"✓ Memory: {total_mem_kb:.1f} KB / {ESP32_SRAM_KB} KB ({total_mem_kb/ESP32_SRAM_KB*100:.1f}%)")


# ── List dataset audio files ─────────────────────────────────────────────────

def get_dataset_files():
    """Get non-augmented audio files from data/raw/."""
    raw_dir = Path("data/raw")
    files = {}
    for cls in CLASSES:
        cls_dir = raw_dir / cls
        if cls_dir.exists():
            cls_files = sorted([
                f.name for f in cls_dir.glob("*.*")
                if f.suffix in {".wav", ".mp3", ".ogg", ".flac"}
                and not f.stem.startswith("aug_")
            ])[:8]  # Limit to 8 per class
            if cls_files:
                files[cls] = cls_files
    return files


@app.route("/")
def index():
    dataset_files = get_dataset_files()
    return render_template(
        "index.html",
        classes=CLASSES,
        thresholds=THRESHOLDS,
        led_colors=LED_COLORS,
        priority=PRIORITY,
        vibration=VIBRATION_PATTERN,
        dataset_files=json.dumps(dataset_files),
        model_size_kb=round(model_size_kb, 1),
        tensor_arena_kb=round(tensor_arena_kb, 1),
        total_mem_kb=round(total_mem_kb, 1),
        sram_kb=ESP32_SRAM_KB,
        flash_mb=ESP32_FLASH_MB,
        clock_mhz=ESP32_CLOCK_MHZ,
        mem_pct=round(total_mem_kb / ESP32_SRAM_KB * 100, 1),
    )


@app.route("/api/signal-pipeline", methods=["POST"])
def signal_pipeline():
    """Return step-by-step signal processing data for visualization."""
    cls = request.form.get("class", "")
    fname = request.form.get("file", "")
    audio_path = Path("data/raw") / cls / fname
    if not audio_path.exists():
        return jsonify({"error": f"File not found: {audio_path}"}), 404

    audio, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True, duration=1.0)
    if len(audio) < WINDOW_SAMPLES:
        audio = np.pad(audio, (0, WINDOW_SAMPLES - len(audio)))
    else:
        audio = audio[:WINDOW_SAMPLES]

    waveform = audio[::4].tolist()

    mel = librosa.feature.melspectrogram(
        y=audio.astype(np.float32), sr=SAMPLE_RATE, n_fft=N_FFT,
        hop_length=HOP_LENGTH, n_mels=N_MELS, fmin=F_MIN, fmax=F_MAX, power=2.0
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    lm_norm = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-8)

    x = lm_norm[np.newaxis, ..., np.newaxis]
    x_q = (x / scale_in + zero_in).astype(np.int8)

    s = 2
    return jsonify({
        "waveform": waveform,
        "rms": round(float(np.sqrt(np.mean(audio ** 2))), 5),
        "samples": len(audio),
        "mel_shape": list(log_mel.shape),
        "log_mel": log_mel[::s, ::s].tolist(),
        "log_mel_min": round(float(log_mel.min()), 1),
        "log_mel_max": round(float(log_mel.max()), 1),
        "normalized": lm_norm[::s, ::s].tolist(),
        "int8": x_q[0, ::s, ::s, 0].tolist(),
        "int8_min": int(x_q.min()),
        "int8_max": int(x_q.max()),
        "quant_scale": round(float(scale_in), 6),
        "quant_zero": int(zero_in),
        "float32_kb": round(float(lm_norm.nbytes / 1024), 1),
        "int8_kb": round(float(x_q.nbytes / 1024), 1),
    })


@app.route("/api/confusion-matrix")
def confusion_matrix_img():
    cm = Path("models/student/confusion_matrix.png")
    if cm.exists():
        from flask import send_file
        return send_file(str(cm.resolve()), mimetype="image/png")
    return jsonify({"error": "not found"}), 404


@app.route("/api/test-file", methods=["POST"])
def test_file():
    """Run inference on an uploaded or dataset audio file, returning per-frame results."""
    # Determine audio source
    if "file" in request.files:
        file = request.files["file"]
        tmp_path = Path("/tmp/safesound_upload.wav")
        file.save(str(tmp_path))
        audio_path = str(tmp_path)
    elif "dataset_file" in request.form:
        cls = request.form.get("dataset_class", "")
        fname = request.form.get("dataset_file", "")
        audio_path = str(Path("data/raw") / cls / fname)
        if not Path(audio_path).exists():
            return jsonify({"error": f"File not found: {audio_path}"}), 404
    else:
        return jsonify({"error": "No audio provided"}), 400

    try:
        audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    except Exception as e:
        return jsonify({"error": f"Could not load audio: {e}"}), 400

    duration = len(audio) / SAMPLE_RATE
    stride = WINDOW_SAMPLES // 2  # 500ms stride

    smooth_buffer = np.ones((SMOOTH_FRAMES, NUM_CLASSES)) / NUM_CLASSES
    smooth_idx = 0
    results = []

    for start in range(0, len(audio) - WINDOW_SAMPLES + 1, stride):
        window = audio[start:start + WINDOW_SAMPLES]
        result, smooth_idx = run_full_inference(
            window, interpreter, inp_details, out_details,
            scale_in, zero_in, scale_out, zero_out,
            smooth_buffer, smooth_idx
        )
        result["time"] = round(start / SAMPLE_RATE, 2)
        result["frame"] = len(results)
        results.append(result)

    return jsonify({
        "duration": round(duration, 2),
        "total_frames": len(results),
        "results": results,
    })


@sock.route("/ws")
def websocket(ws):
    """Handle live audio stream from browser mic."""
    # Per-connection smoothing buffer
    smooth_buffer = np.ones((SMOOTH_FRAMES, NUM_CLASSES)) / NUM_CLASSES
    smooth_idx = 0

    while True:
        try:
            data = ws.receive()
            if data is None:
                break

            # Decode binary Float32 PCM audio
            if isinstance(data, bytes):
                n_samples = len(data) // 4
                audio = np.array(struct.unpack(f"{n_samples}f", data), dtype=np.float32)
            else:
                # JSON control message
                try:
                    msg = json.loads(data)
                    if msg.get("type") == "reset":
                        smooth_buffer = np.ones((SMOOTH_FRAMES, NUM_CLASSES)) / NUM_CLASSES
                        smooth_idx = 0
                        ws.send(json.dumps({"type": "reset_ack"}))
                    continue
                except (json.JSONDecodeError, TypeError):
                    continue

            if len(audio) < SAMPLE_RATE // 2:
                continue

            # Pad or trim to exactly 1 second
            if len(audio) < WINDOW_SAMPLES:
                audio = np.pad(audio, (0, WINDOW_SAMPLES - len(audio)))
            else:
                audio = audio[:WINDOW_SAMPLES]

            result, smooth_idx = run_full_inference(
                audio, interpreter, inp_details, out_details,
                scale_in, zero_in, scale_out, zero_out,
                smooth_buffer, smooth_idx
            )
            ws.send(json.dumps(result))

        except Exception as e:
            print(f"WebSocket error: {e}")
            break


if __name__ == "__main__":
    print("\n" + "=" * 56)
    print("  SafeSound — Live Sound Alert Dashboard")
    print("  Open http://localhost:5050 in your browser")
    print("=" * 56 + "\n")
    app.run(host="0.0.0.0", port=5050, debug=False)
