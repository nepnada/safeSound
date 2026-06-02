"""
Sound Alert System — Professional Demo
=======================================
Uses the real trained DS-CNN INT8 TFLite model.
Upload any audio file OR record from your microphone.
The model classifies it and shows which alert would trigger on the bracelet.

Run: python demo.py
"""

import gradio as gr
import numpy as np
import tensorflow as tf
import librosa
import time
from pathlib import Path

# ── Model config ──────────────────────────────────────────────────────────────
TFLITE_PATH = "models/student/dscnn_int8.tflite"
SAMPLE_RATE  = 16000
N_MELS, N_FFT, HOP_LENGTH = 64, 512, 256
CLASSES = ["🔥 Fire Alarm", "👶 Baby Crying", "😮 Choking", "🚗 Car Horn", "🔔 Doorbell", "🔇 Background"]
CLASS_KEYS = ["fire_alarm", "baby_cry", "choking", "car_horn", "doorbell", "background"]
THRESHOLDS  = [0.75, 0.80, 0.75, 0.80, 0.90, 0.50]
COLORS      = ["#FF2200", "#FF8C00", "#9B00FF", "#FFD700", "#0066FF", "#888888"]
CRITICAL    = {0, 2}   # fire_alarm, choking → critical alert

# Load model once
interp = tf.lite.Interpreter(model_path=TFLITE_PATH)
interp.allocate_tensors()
inp_det = interp.get_input_details()[0]
out_det = interp.get_output_details()[0]
s_in, z_in   = inp_det["quantization"]
s_out, z_out = out_det["quantization"]


def infer(audio: np.ndarray) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=audio, sr=SAMPLE_RATE, n_fft=N_FFT,
        hop_length=HOP_LENGTH, n_mels=N_MELS, fmin=60, fmax=7600, power=2.0)
    lm = librosa.power_to_db(mel, ref=np.max)
    lm = (lm - lm.min()) / (lm.max() - lm.min() + 1e-8)
    x  = lm[np.newaxis, ..., np.newaxis].astype(np.float32)
    xq = (x / s_in + z_in).astype(np.int8)
    interp.set_tensor(inp_det["index"], xq)
    interp.invoke()
    yq    = interp.get_tensor(out_det["index"])
    probs = (yq.astype(np.float32) - z_out) * s_out
    return probs[0]


def classify_audio(audio_input):
    """Main inference function called by Gradio."""
    if audio_input is None:
        return (
            gr.update(value="⬆️  Upload an audio file or record from microphone"),
            {c: 0.0 for c in CLASSES},
            gr.update(value="<div style='text-align:center;padding:20px;color:#888'>No audio</div>")
        )

    sr, data = audio_input
    audio = data.astype(np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio / (np.max(np.abs(audio)) + 1e-8)

    # Resample if needed
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)

    # Sliding window inference (1s windows, 0.5s stride)
    window = SAMPLE_RATE
    stride = SAMPLE_RATE // 2
    smooth = np.ones((3, 6)) / 6
    sidx   = 0

    for start in range(0, max(1, len(audio) - window + 1), stride):
        chunk = audio[start:start + window]
        if len(chunk) < window:
            chunk = np.pad(chunk, (0, window - len(chunk)))
        probs       = infer(chunk)
        smooth[sidx] = probs
        sidx         = (sidx + 1) % 3

    avg  = smooth.mean(axis=0)
    best = int(np.argmax(avg))
    conf = float(avg[best])
    alert_triggered = conf >= THRESHOLDS[best] and best != 5

    # ── Confidence dict for Gradio BarPlot ───────────────────────────────────
    conf_dict = {CLASSES[i]: float(avg[i]) for i in range(6)}

    # ── Result HTML ──────────────────────────────────────────────────────────
    color  = COLORS[best]
    cls_name = CLASS_KEYS[best]
    is_crit  = best in CRITICAL
    pulse    = "animation: pulse 0.5s ease-in-out infinite alternate;" if alert_triggered else ""

    if alert_triggered:
        vibro_pattern = "●●●●● CRITICAL" if is_crit else "●●● NORMAL"
        vibro_color   = "#FF4444" if is_crit else "#FF8C00"
        alert_box = f"""
        <div style="
            background: {color}22;
            border: 3px solid {color};
            border-radius: 16px;
            padding: 24px;
            text-align: center;
            {pulse}
            box-shadow: 0 0 30px {color}66;
        ">
            <div style="font-size: 64px; margin-bottom: 8px;">
                {'🚨' if is_crit else '⚠️'}
            </div>
            <div style="font-size: 28px; font-weight: bold; color: {color}; letter-spacing: 2px;">
                ALERT TRIGGERED
            </div>
            <div style="font-size: 22px; margin: 12px 0; color: white; font-weight: bold;">
                {CLASSES[best]}
            </div>
            <div style="font-size: 18px; color: #ccc; margin-bottom: 16px;">
                Confidence: <b style="color:{color}">{conf:.1%}</b>
                &nbsp;|&nbsp; Threshold: {THRESHOLDS[best]:.0%}
            </div>
            <div style="display: flex; justify-content: center; gap: 16px; flex-wrap: wrap;">
                <div style="background: {color}33; border: 1px solid {color}; border-radius: 8px; padding: 10px 20px;">
                    <div style="font-size: 11px; color: #aaa; text-transform: uppercase;">LED Color</div>
                    <div style="width: 40px; height: 20px; background: {color}; border-radius: 4px; margin: 4px auto;
                         box-shadow: 0 0 12px {color};"></div>
                </div>
                <div style="background: {vibro_color}33; border: 1px solid {vibro_color}; border-radius: 8px; padding: 10px 20px;">
                    <div style="font-size: 11px; color: #aaa; text-transform: uppercase;">Vibration</div>
                    <div style="color: {vibro_color}; font-weight: bold; font-size: 14px;">{vibro_pattern}</div>
                </div>
                <div style="background: #33333388; border: 1px solid #555; border-radius: 8px; padding: 10px 20px;">
                    <div style="font-size: 11px; color: #aaa; text-transform: uppercase;">Latency</div>
                    <div style="color: #eee; font-weight: bold; font-size: 14px;">~2ms (ESP32)</div>
                </div>
            </div>
        </div>"""
    else:
        alert_box = f"""
        <div style="
            background: #1a1a2e;
            border: 2px solid #333;
            border-radius: 16px;
            padding: 24px;
            text-align: center;
        ">
            <div style="font-size: 48px;">🔇</div>
            <div style="font-size: 22px; color: #888; margin: 8px 0;">No Alert</div>
            <div style="color: #555; font-size: 14px;">
                {CLASSES[best]} detected at {conf:.1%} — below threshold ({THRESHOLDS[best]:.0%})<br>
                LEDs: OFF &nbsp;|&nbsp; Vibration: none
            </div>
        </div>"""

    # Status line
    status = f"✅ Analyzed — {CLASSES[best]} ({conf:.1%}) | Model: DS-CNN INT8 36.4KB | Frames: {max(1, (len(audio)-SAMPLE_RATE)//stride+1)}"

    return (
        gr.update(value=status),
        conf_dict,
        gr.update(value=alert_box)
    )


# ── CSS ───────────────────────────────────────────────────────────────────────
CSS = """
body, .gradio-container { background: #0d0d1a !important; }
.gr-box, .gr-form { background: #1a1a2e !important; border-color: #333 !important; }
h1 { color: #eee !important; }
.gr-button-primary { background: #3498db !important; border: none !important; }
footer { display: none !important; }
"""

# ── UI ────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="Sound Alert System") as demo:

    gr.HTML("""
    <div style="text-align:center; padding: 24px 0 12px;">
        <h1 style="font-size:2em; margin:0; color:#eee;">
            🎧 Sound Alert System
        </h1>
        <p style="color:#888; margin:8px 0 0;">
            Wearable IoT device for the deaf/hard-of-hearing &nbsp;·&nbsp;
            <b style="color:#3498db">DS-CNN INT8</b> &nbsp;·&nbsp;
            <b style="color:#3498db">36.4 KB</b> &nbsp;·&nbsp;
            <b style="color:#3498db">18,630 params</b> &nbsp;·&nbsp;
            ESP32-S3 TFLite Micro
        </p>
    </div>
    """)

    with gr.Row():
        # ── Left column: input ───────────────────────────────────────────────
        with gr.Column(scale=1):
            gr.HTML("<h3 style='color:#eee; margin-bottom:4px;'>🎙️ Audio Input</h3>")
            audio_in = gr.Audio(
                label="Upload file or record from microphone",
                sources=["upload", "microphone"],
                type="numpy",
            )
            run_btn = gr.Button("▶  Classify", variant="primary", size="lg")

            gr.HTML("""
            <div style="background:#1a1a2e; border:1px solid #333; border-radius:8px; padding:14px; margin-top:8px;">
                <div style="color:#888; font-size:12px; margin-bottom:8px; text-transform:uppercase; letter-spacing:1px;">
                    Test Sounds — Try These
                </div>
                <div style="display:flex; flex-wrap:wrap; gap:6px;">
                    <span style="background:#FF220022; border:1px solid #FF2200; color:#FF2200; padding:3px 10px; border-radius:4px; font-size:12px;">🔥 Fire alarm .wav</span>
                    <span style="background:#FF8C0022; border:1px solid #FF8C00; color:#FF8C00; padding:3px 10px; border-radius:4px; font-size:12px;">👶 Baby cry .wav</span>
                    <span style="background:#FFD70022; border:1px solid #FFD700; color:#FFD700; padding:3px 10px; border-radius:4px; font-size:12px;">🚗 Car horn .wav</span>
                    <span style="background:#0066FF22; border:1px solid #0066FF; color:#0066FF; padding:3px 10px; border-radius:4px; font-size:12px;">🔔 Doorbell .wav</span>
                </div>
                <div style="color:#555; font-size:11px; margin-top:8px;">
                    Find samples in <code style="color:#888">data/raw/&lt;class&gt;/</code>
                </div>
            </div>
            """)

            gr.HTML("<h3 style='color:#eee; margin:16px 0 4px;'>📊 Confidence per Class</h3>")
            bar = gr.Label(label="", num_top_classes=6)

        # ── Right column: result ─────────────────────────────────────────────
        with gr.Column(scale=1):
            gr.HTML("<h3 style='color:#eee; margin-bottom:4px;'>⌚ Bracelet Output</h3>")
            result_html = gr.HTML(
                value="<div style='text-align:center;padding:40px;color:#555;"
                      "border:2px dashed #333;border-radius:16px;font-size:16px;'>"
                      "Upload or record audio to see the alert output</div>"
            )

            # Hardware info
            gr.HTML("""
            <div style="background:#1a1a2e; border:1px solid #333; border-radius:8px;
                        padding:14px; margin-top:12px;">
                <div style="color:#888; font-size:12px; margin-bottom:10px;
                            text-transform:uppercase; letter-spacing:1px;">Hardware</div>
                <div style="display:grid; grid-template-columns:1fr 1fr; gap:6px; font-size:12px;">
                    <div style="color:#aaa;">🔲 MCU</div>
                    <div style="color:#3498db;">ESP32-S3 240MHz</div>
                    <div style="color:#aaa;">🎤 Microphone</div>
                    <div style="color:#3498db;">INMP441 I2S</div>
                    <div style="color:#aaa;">💡 LEDs</div>
                    <div style="color:#3498db;">WS2812B ×8 RGB</div>
                    <div style="color:#aaa;">📳 Haptic</div>
                    <div style="color:#3498db;">ERM Motor</div>
                    <div style="color:#aaa;">🔋 Power</div>
                    <div style="color:#3498db;">LiPo 3.7V 1000mAh</div>
                    <div style="color:#aaa;">⚡ Latency</div>
                    <div style="color:#2ecc71;">~2ms on device</div>
                </div>
            </div>
            """)

    # Status bar
    status_txt = gr.Textbox(
        label="", value="⬆️  Upload an audio file or record from your microphone",
        interactive=False, container=False
    )

    # ── Confidence thresholds legend ─────────────────────────────────────────
    with gr.Accordion("ℹ️  About the model & thresholds", open=False):
        gr.HTML("""
        <div style="padding:12px; font-size:13px; color:#aaa;">
        <b style="color:#eee">Architecture:</b> DS-CNN (Depthwise Separable CNN)
        trained via <b style="color:#3498db">Knowledge Distillation</b> from a larger teacher model.<br><br>
        <b style="color:#eee">Pipeline:</b>
        16kHz audio → Log-Mel Spectrogram (64ch) → DS-CNN INT8 → Temporal Smoothing → Alert<br><br>
        <div style="display:grid; grid-template-columns:repeat(3,1fr); gap:8px; margin-top:8px;">
            <div style="background:#FF220011; border:1px solid #FF220066; border-radius:6px; padding:8px;">
                <b style="color:#FF2200">🔥 Fire Alarm</b><br>
                <span style="color:#888">Threshold: 75% — Critical</span>
            </div>
            <div style="background:#FF8C0011; border:1px solid #FF8C0066; border-radius:6px; padding:8px;">
                <b style="color:#FF8C00">👶 Baby Crying</b><br>
                <span style="color:#888">Threshold: 80%</span>
            </div>
            <div style="background:#9B00FF11; border:1px solid #9B00FF66; border-radius:6px; padding:8px;">
                <b style="color:#9B00FF">😮 Choking</b><br>
                <span style="color:#888">Threshold: 75% — Critical</span>
            </div>
            <div style="background:#FFD70011; border:1px solid #FFD70066; border-radius:6px; padding:8px;">
                <b style="color:#FFD700">🚗 Car Horn</b><br>
                <span style="color:#888">Threshold: 80%</span>
            </div>
            <div style="background:#0066FF11; border:1px solid #0066FF66; border-radius:6px; padding:8px;">
                <b style="color:#0066FF">🔔 Doorbell</b><br>
                <span style="color:#888">Threshold: 90%</span>
            </div>
            <div style="background:#88888811; border:1px solid #88888866; border-radius:6px; padding:8px;">
                <b style="color:#888">🔇 Background</b><br>
                <span style="color:#888">No alert triggered</span>
            </div>
        </div>
        </div>
        """)

    # ── Circuit & Hardware Section ────────────────────────────────────────────
    gr.HTML("<hr style='border-color:#333; margin: 24px 0;'>")
    gr.HTML("<h2 style='color:#eee; text-align:center; margin-bottom:4px;'>🔌 Hardware — ESP32-S3 Circuit</h2>")
    gr.HTML("<p style='color:#888; text-align:center; margin-bottom:16px;'>Full wiring diagram: INMP441 microphone · WS2812B RGB LEDs · ERM vibration motor · LiPo battery</p>")

    with gr.Row():
        with gr.Column(scale=2):
            gr.Image(
                value="notebooks/figures/fritzing_breadboard.png",
                label="Breadboard Wiring Diagram",
                interactive=False,
                
            )
        with gr.Column(scale=1):
            gr.Image(
                value="notebooks/figures/circuit_schematic.png",
                label="System Architecture",
                interactive=False,
                
            )
            gr.HTML("""
            <div style="background:#1a1a2e; border:1px solid #333; border-radius:8px; padding:14px; margin-top:8px; font-size:12px;">
                <b style="color:#eee;">Wire color legend:</b>
                <div style="margin-top:8px; display:grid; grid-template-columns:16px 1fr; gap:4px 8px; align-items:center;">
                    <div style="background:#FF4444; height:4px; border-radius:2px;"></div><span style="color:#aaa;">VCC / 3.3V</span>
                    <div style="background:#222; height:4px; border-radius:2px; border:1px solid #555;"></div><span style="color:#aaa;">GND</span>
                    <div style="background:#00CC44; height:4px; border-radius:2px;"></div><span style="color:#aaa;">NeoPixel Data (GPIO 5)</span>
                    <div style="background:#FF8800; height:4px; border-radius:2px;"></div><span style="color:#aaa;">I2S Clock / Motor PWM</span>
                    <div style="background:#0088FF; height:4px; border-radius:2px;"></div><span style="color:#aaa;">I2S Word Select</span>
                    <div style="background:#8800FF; height:4px; border-radius:2px;"></div><span style="color:#aaa;">Button / Status LED</span>
                </div>
                <div style="margin-top:12px; padding-top:10px; border-top:1px solid #333;">
                    <b style="color:#eee;">Wokwi Simulation:</b><br>
                    <a href="https://wokwi.com/projects/465708495578640385"
                       target="_blank"
                       style="color:#3498db; font-size:12px;">
                        🔗 wokwi.com/projects/465708495578640385
                    </a><br>
                    <span style="color:#555; font-size:11px;">Interactive circuit — press TEST CLASS button to simulate detection</span>
                </div>
            </div>
            """)

    # ── Results figures ────────────────────────────────────────────────────────
    gr.HTML("<hr style='border-color:#333; margin: 24px 0;'>")
    gr.HTML("<h2 style='color:#eee; text-align:center; margin-bottom:16px;'>📈 Model Performance</h2>")
    with gr.Row():
        gr.Image(value="notebooks/figures/f1_per_class.png",      label="F1 Score per Class",    interactive=False)
        gr.Image(value="notebooks/figures/spectrograms.png",      label="Log-Mel Spectrograms",  interactive=False)
        gr.Image(value="notebooks/figures/model_comparison.png",  label="Teacher vs Student",    interactive=False)

    # Events
    run_btn.click(
        fn=classify_audio,
        inputs=[audio_in],
        outputs=[status_txt, bar, result_html]
    )
    audio_in.change(
        fn=classify_audio,
        inputs=[audio_in],
        outputs=[status_txt, bar, result_html]
    )


if __name__ == "__main__":
    demo.launch(server_port=7860, share=False, css=CSS, theme=gr.themes.Base())
