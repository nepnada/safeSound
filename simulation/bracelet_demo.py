"""
Sound Alert Bracelet — Interactive Demo
=========================================
Simulates the full wearable system visually:
- Animated bracelet with RGB LEDs
- Audio playback of real sound samples
- Live model inference (real TFLite INT8)
- Vibration motor animation
- Confidence bars per class

Run: streamlit run simulation/bracelet_demo.py
"""

import streamlit as st
import numpy as np
import tensorflow as tf
import librosa
import time
import soundfile as sf
import io
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
SAMPLE_RATE = 16000
N_MELS = 64
N_FFT = 512
HOP_LENGTH = 256
CLASSES = ["fire_alarm", "baby_cry", "choking", "car_horn", "doorbell", "background"]
THRESHOLDS = [0.75, 0.80, 0.75, 0.80, 0.90, 0.50]
LED_HEX = {
    "fire_alarm": "#FF2200",
    "baby_cry":   "#FF8C00",
    "choking":    "#9B00FF",
    "car_horn":   "#FFD700",
    "doorbell":   "#0066FF",
    "background": "#1a1a2e",
}
CLASS_ICONS = {
    "fire_alarm": "🔥",
    "baby_cry":   "👶",
    "choking":    "😮",
    "car_horn":   "🚗",
    "doorbell":   "🔔",
    "background": "🔇",
}
TFLITE_PATH = Path("models/student/dscnn_int8.tflite")


@st.cache_resource
def load_model():
    interp = tf.lite.Interpreter(model_path=str(TFLITE_PATH))
    interp.allocate_tensors()
    return interp


def compute_logmel(audio):
    mel = librosa.feature.melspectrogram(
        y=audio.astype(np.float32), sr=SAMPLE_RATE, n_fft=N_FFT,
        hop_length=HOP_LENGTH, n_mels=N_MELS, fmin=60.0, fmax=7600.0, power=2.0
    )
    lm = librosa.power_to_db(mel, ref=np.max)
    lm = (lm - lm.min()) / (lm.max() - lm.min() + 1e-8)
    return lm.astype(np.float32)


def run_inference(interp, audio):
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    s_in, z_in = inp["quantization"]
    s_out, z_out = out["quantization"]
    lm = compute_logmel(audio)
    x = lm[np.newaxis, ..., np.newaxis]
    x_q = (x / s_in + z_in).astype(np.int8)
    t0 = time.perf_counter()
    interp.set_tensor(inp["index"], x_q)
    interp.invoke()
    latency = (time.perf_counter() - t0) * 1000
    y_q = interp.get_tensor(out["index"])
    probs = (y_q.astype(np.float32) - z_out) * s_out
    return probs[0], latency


def render_bracelet(active_class: str, confidence: float, alert: bool, vibrating: bool):
    """Renders animated bracelet HTML/CSS."""
    led_color = LED_HEX.get(active_class, "#1a1a2e")
    glow = f"0 0 20px {led_color}, 0 0 40px {led_color}88" if alert else "none"
    pulse_anim = "pulse 0.4s ease-in-out infinite alternate" if vibrating else "none"
    led_opacity = "1" if alert else "0.15"

    # 8 LEDs, active ones show the class color
    leds_html = ""
    for i in range(8):
        if alert:
            color = led_color
            glow_led = f"0 0 12px {led_color}"
            opacity = "1"
        else:
            color = "#333"
            glow_led = "none"
            opacity = "0.3"
        leds_html += f"""
        <div style="
            width:14px; height:14px; border-radius:50%;
            background:{color};
            box-shadow:{glow_led};
            opacity:{opacity};
            transition: all 0.3s ease;
            margin:2px;
        "></div>"""

    vibro_color = "#FF6B00" if vibrating else "#444"
    vibro_anim = "vibro 0.15s ease-in-out infinite" if vibrating else "none"

    html = f"""
    <style>
        @keyframes pulse {{
            from {{ transform: scale(1); box-shadow: {glow}; }}
            to {{ transform: scale(1.03); box-shadow: {glow}, 0 0 60px {led_color}44; }}
        }}
        @keyframes vibro {{
            0%   {{ transform: rotate(-2deg) translateX(-2px); }}
            50%  {{ transform: rotate(2deg) translateX(2px); }}
            100% {{ transform: rotate(-2deg) translateX(-2px); }}
        }}
        @keyframes blink {{
            0%,100% {{ opacity:1; }}
            50% {{ opacity:0.3; }}
        }}
    </style>

    <div style="display:flex; flex-direction:column; align-items:center; padding:30px;">

        <!-- Bracelet body -->
        <div style="
            animation: {pulse_anim};
            background: linear-gradient(135deg, #2c2c3e 0%, #1a1a2e 100%);
            border: 2px solid {led_color if alert else '#444'};
            border-radius: 20px;
            padding: 20px 30px;
            width: 280px;
            box-shadow: {glow};
            transition: all 0.3s ease;
            position: relative;
        ">
            <!-- Chip label -->
            <div style="
                background: #0d0d1a;
                border-radius: 8px;
                padding: 6px 12px;
                margin-bottom: 15px;
                text-align: center;
                border: 1px solid #333;
            ">
                <span style="color:#666; font-size:10px; font-family:monospace;">ESP32-S3 · DS-CNN INT8 · 36.4KB</span>
            </div>

            <!-- LED strip -->
            <div style="display:flex; justify-content:center; margin-bottom:15px; flex-wrap:wrap;">
                {leds_html}
            </div>

            <!-- Class display -->
            <div style="text-align:center; margin-bottom:12px;">
                <div style="font-size:32px;">{CLASS_ICONS.get(active_class,'❓')}</div>
                <div style="
                    color: {led_color if alert else '#666'};
                    font-size:16px; font-weight:bold;
                    font-family: monospace;
                    text-transform: uppercase;
                    letter-spacing: 2px;
                    margin-top:4px;
                ">{active_class.replace('_',' ')}</div>
                <div style="
                    color:#888; font-size:12px; margin-top:2px;
                ">conf: {confidence:.1%}</div>
            </div>

            <!-- Vibration motor indicator -->
            <div style="
                animation: {vibro_anim};
                background: {vibro_color};
                border-radius: 8px;
                padding: 5px 10px;
                text-align:center;
                font-size:11px;
                color: white;
                font-family: monospace;
                transition: background 0.2s ease;
            ">
                {'⚡ VIBRATING' if vibrating else '○ motor idle'}
            </div>

            <!-- Alert badge -->
            {'<div style="position:absolute; top:-10px; right:-10px; background:#FF2200; color:white; border-radius:50%; width:26px; height:26px; display:flex; align-items:center; justify-content:center; font-size:14px; animation:blink 0.5s ease infinite;">🚨</div>' if alert else ''}
        </div>

        <!-- Strap -->
        <div style="
            width:40px; height:20px;
            background: linear-gradient(180deg, #2c2c3e, #1a1a2e);
            border-left: 2px solid #333;
            border-right: 2px solid #333;
            border-bottom: 2px solid #333;
            border-radius: 0 0 8px 8px;
        "></div>
    </div>
    """
    return html


# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sound Alert Bracelet Demo",
    page_icon="⌚",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom dark theme
st.markdown("""
<style>
    .stApp { background: #0a0a12; color: #eee; }
    .block-container { padding-top: 1rem; }
    h1, h2, h3 { color: #eee !important; }
    .stMetric label { color: #aaa !important; }
    .stMetric [data-testid="metric-container"] { background: #1a1a2e; border-radius: 8px; padding: 10px; }
</style>
""", unsafe_allow_html=True)

st.markdown("# ⌚ Sound Alert Bracelet — Live Demo")
st.markdown("*Wearable IoT device for the deaf/hard-of-hearing · ESP32-S3 + DS-CNN INT8*")
st.divider()

# ── Load model ─────────────────────────────────────────────────────────────
if not TFLITE_PATH.exists():
    st.error("Model not found. Run: bash run_pipeline.sh")
    st.stop()

interp = load_model()

# ── Layout ─────────────────────────────────────────────────────────────────
col_bracelet, col_demo, col_metrics = st.columns([1.2, 2, 1.5])

# ── Bracelet column ─────────────────────────────────────────────────────────
with col_bracelet:
    st.markdown("### 📟 Bracelet")
    bracelet_slot = st.empty()
    # Default state
    bracelet_slot.markdown(
        render_bracelet("background", 0.0, False, False),
        unsafe_allow_html=True
    )

# ── Demo column ─────────────────────────────────────────────────────────────
with col_demo:
    st.markdown("### 🎵 Audio Input")

    RAW_DIR = Path("data/raw")
    source_mode = st.radio("Source:", ["Dataset samples", "Upload audio"], horizontal=True)

    if source_mode == "Dataset samples":
        available = {}
        for cls in CLASSES:
            d = RAW_DIR / cls
            if d.exists():
                files = sorted([f for f in d.glob("*.*")
                    if f.suffix in {".wav",".mp3",".ogg"} and not f.stem.startswith("aug_")])
                if files:
                    available[cls] = files[:8]

        c1, c2 = st.columns(2)
        with c1:
            sel_cls = st.selectbox("Class", list(available.keys()),
                                   format_func=lambda c: f"{CLASS_ICONS[c]} {c}")
        with c2:
            sel_file = st.selectbox("File", available.get(sel_cls, []),
                                    format_func=lambda f: f.name)
        audio_path = str(sel_file) if sel_cls in available else None

    else:
        uploaded = st.file_uploader("Upload .wav/.mp3", type=["wav","mp3"])
        if uploaded:
            tmp = Path("/tmp/demo_upload.wav")
            tmp.write_bytes(uploaded.read())
            audio_path = str(tmp)
        else:
            audio_path = None

    run_btn = st.button("▶  Run Inference", type="primary", use_container_width=True,
                        disabled=audio_path is None)

    # Progress + results area
    progress_slot = st.empty()
    waveform_slot = st.empty()
    confidence_slot = st.empty()

# ── Metrics column ──────────────────────────────────────────────────────────
with col_metrics:
    st.markdown("### 📊 Model Info")
    st.metric("Model size", "36.4 KB")
    st.metric("Parameters", "18,630")
    st.metric("Quantization", "INT8")
    st.metric("Architecture", "DS-CNN")
    st.divider()
    st.markdown("### 🎯 Thresholds")
    for i, cls in enumerate(CLASSES):
        st.markdown(
            f'<div style="display:flex;justify-content:space-between;'
            f'background:#1a1a2e;padding:4px 8px;border-radius:4px;margin:2px 0;">'
            f'<span style="color:#aaa;font-size:12px;">{CLASS_ICONS[cls]} {cls}</span>'
            f'<span style="color:{LED_HEX[cls]};font-size:12px;font-weight:bold;">'
            f'{THRESHOLDS[i]:.0%}</span></div>',
            unsafe_allow_html=True
        )
    latency_slot = st.empty()

# ── Run inference ────────────────────────────────────────────────────────────
if run_btn and audio_path:
    audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    stride = SAMPLE_RATE // 2
    n_frames = max(1, (len(audio) - SAMPLE_RATE) // stride + 1)

    smooth_buf = np.ones((3, 6)) / 6
    s_idx = 0
    all_probs = []
    alert_history = []
    latencies = []

    # Play audio in browser
    audio_bytes = io.BytesIO()
    sf.write(audio_bytes, audio, SAMPLE_RATE, format='WAV')
    audio_bytes.seek(0)
    with col_demo:
        st.audio(audio_bytes, format="audio/wav")

    prog = progress_slot.progress(0, text="Running inference...")

    for i, start in enumerate(range(0, len(audio) - SAMPLE_RATE + 1, stride)):
        window = audio[start:start + SAMPLE_RATE]
        probs, latency = run_inference(interp, window)
        latencies.append(latency)

        smooth_buf[s_idx] = probs
        s_idx = (s_idx + 1) % 3
        avg = smooth_buf.mean(axis=0)

        best = int(np.argmax(avg))
        cls_name = CLASSES[best]
        conf = float(avg[best])
        alert = conf >= THRESHOLDS[best] and best != 5

        all_probs.append(avg.copy())
        alert_history.append(alert)

        # Update bracelet
        bracelet_slot.markdown(
            render_bracelet(cls_name, conf, alert, alert and best in [0,2]),
            unsafe_allow_html=True
        )

        # Update confidence bars
        bars_html = '<div style="font-family:monospace;">'
        for j, c in enumerate(CLASSES):
            p = avg[j]
            bar_w = int(p * 200)
            color = LED_HEX[c]
            active_mark = "◀" if j == best else ""
            alert_mark = " 🚨" if j == best and alert else ""
            bars_html += f"""
            <div style="margin:3px 0;display:flex;align-items:center;gap:8px;">
                <span style="width:80px;font-size:11px;color:#aaa;text-align:right;">{CLASS_ICONS[c]} {c}</span>
                <div style="background:#222;border-radius:3px;width:200px;height:14px;overflow:hidden;">
                    <div style="background:{color};width:{bar_w}px;height:100%;
                        border-radius:3px;transition:width 0.2s ease;
                        {'box-shadow:0 0 8px ' + color if alert and j==best else ''}">
                    </div>
                </div>
                <span style="color:{color};font-size:11px;width:45px;">{p:.1%}{active_mark}{alert_mark}</span>
            </div>"""
        bars_html += f"<div style='color:#666;font-size:10px;margin-top:8px;'>Frame {i+1}/{n_frames} · {latency*8:.1f}ms (ESP32 est.)</div>"
        bars_html += '</div>'
        confidence_slot.markdown(bars_html, unsafe_allow_html=True)

        prog.progress((i+1)/n_frames, text=f"Frame {i+1}/{n_frames}")
        time.sleep(0.1)  # pacing for visual effect

    prog.empty()

    # Final summary
    final_cls = CLASSES[int(np.argmax(all_probs[-1]))]
    n_alerts = sum(alert_history)
    avg_lat = np.mean(latencies) * 8

    latency_slot.metric("Avg latency (ESP32)", f"{avg_lat:.1f}ms")

    with col_demo:
        st.divider()
        if n_alerts > 0:
            st.markdown(
                f'<div style="background:{LED_HEX[final_cls]}22;border:2px solid {LED_HEX[final_cls]};'
                f'border-radius:10px;padding:15px;text-align:center;">'
                f'<div style="font-size:40px;">{CLASS_ICONS[final_cls]}</div>'
                f'<div style="font-size:20px;font-weight:bold;color:{LED_HEX[final_cls]};">'
                f'🚨 {final_cls.replace("_"," ").upper()} DETECTED</div>'
                f'<div style="color:#aaa;margin-top:5px;">{n_alerts} alert frames out of {n_frames}</div>'
                f'</div>',
                unsafe_allow_html=True
            )
        else:
            st.info("No alert detected — background noise")
