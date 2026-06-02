"""
Live inference test using your Mac microphone.
Captures 1s windows and runs the TFLite model in real-time.
Great for demonstrating the system without the ESP32.

Usage: python src/models/inference_test.py
Press Ctrl+C to stop.

Requirements: pip install sounddevice
"""

import numpy as np
import tensorflow as tf
import time
import sys

try:
    import sounddevice as sd
except ImportError:
    print("Install sounddevice: pip install sounddevice")
    sys.exit(1)

SAMPLE_RATE = 16000
WINDOW = 16000
CLASSES = ["fire_alarm", "baby_cry", "choking", "car_horn", "doorbell", "background"]
TFLITE_PATH = "models/student/dscnn_int8.tflite"
THRESHOLDS = [0.75, 0.80, 0.75, 0.80, 0.90, 0.50]

# Log-mel params
N_MELS = 64
N_FFT = 512
HOP_LENGTH = 256

COLORS = {
    "fire_alarm": "\033[91m",   # red
    "baby_cry": "\033[93m",     # yellow
    "choking": "\033[95m",      # magenta
    "car_horn": "\033[93m",     # yellow
    "doorbell": "\033[94m",     # blue
    "background": "\033[90m",   # gray
}
RESET = "\033[0m"


def audio_to_logmel(audio):
    import librosa
    mel = librosa.feature.melspectrogram(
        y=audio, sr=SAMPLE_RATE, n_fft=N_FFT,
        hop_length=HOP_LENGTH, n_mels=N_MELS,
        fmin=60.0, fmax=7600.0, power=2.0
    )
    lm = librosa.power_to_db(mel, ref=np.max)
    lm = (lm - lm.min()) / (lm.max() - lm.min() + 1e-8)
    return lm.astype(np.float32)[..., np.newaxis][np.newaxis]  # (1, 64, T, 1)


def run():
    if not __import__("pathlib").Path(TFLITE_PATH).exists():
        print(f"Model not found: {TFLITE_PATH}")
        print("Run training first: bash run_pipeline.sh")
        return

    interp = tf.lite.Interpreter(model_path=TFLITE_PATH)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    s_in, z_in = inp["quantization"]
    s_out, z_out = out["quantization"]

    # Smoothing buffer
    smooth = np.ones((3, 6)) / 6

    print(f"\033[1mSound Alert System — Live Test\033[0m")
    print(f"Listening on default microphone at {SAMPLE_RATE}Hz...")
    print("Press Ctrl+C to stop\n")

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                            dtype="float32", blocksize=WINDOW) as stream:
            i = 0
            while True:
                audio, _ = stream.read(WINDOW)
                audio = audio.flatten()

                t0 = time.perf_counter()
                lm = audio_to_logmel(audio)
                x_q = (lm / s_in + z_in).astype(np.int8)
                interp.set_tensor(inp["index"], x_q)
                interp.invoke()
                y_q = interp.get_tensor(out["index"])
                probs = (y_q.astype(np.float32) - z_out) * s_out
                latency = (time.perf_counter() - t0) * 1000

                smooth = np.roll(smooth, -1, axis=0)
                smooth[-1] = probs[0]
                avg = smooth.mean(axis=0)

                best = np.argmax(avg)
                cls = CLASSES[best]
                conf = avg[best]
                alert = conf >= THRESHOLDS[best] and best != 5

                color = COLORS[cls]
                bar = "█" * int(conf * 20)
                alert_str = " 🚨 ALERT" if alert else ""
                print(f"\r{color}[{latency:5.1f}ms] {cls:12s} {conf:.2f} |{bar:<20}|{RESET}{alert_str}   ",
                      end="", flush=True)
                i += 1

    except KeyboardInterrupt:
        print("\n\nStopped.")


if __name__ == "__main__":
    run()
