"""
Real-time Sound Alert Dashboard (Streamlit)
=============================================
Interactive web dashboard that simulates the full ESP32-S3 pipeline.
Shows live audio classification, confidence bars, LED simulation,
and temporal smoothing visualization.

Usage:
    streamlit run simulation/dashboard.py
"""

import streamlit as st
import numpy as np
import tensorflow as tf
import librosa
import time
import json
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px

# ── Config ────────────────────────────────────────────────────────────────────
SAMPLE_RATE = 16000
N_MELS = 64
N_FFT = 512
HOP_LENGTH = 256
NUM_CLASSES = 6
CLASSES = ["fire_alarm", "baby_cry", "choking", "car_horn", "doorbell", "background"]
COLORS = ["#e74c3c", "#f39c12", "#9b59b6", "#f1c40f", "#3498db", "#95a5a6"]
LED_COLORS_HEX = {
    "fire_alarm": "#FF0000",
    "baby_cry":   "#FF8C00",
    "choking":    "#800080",
    "car_horn":   "#FFD700",
    "doorbell":   "#0000FF",
    "background": "#333333",
}
THRESHOLDS = [0.75, 0.80, 0.75, 0.80, 0.90, 0.50]
TFLITE_PATH = Path("models/student/dscnn_int8.tflite")


@st.cache_resource
def load_model():
    interp = tf.lite.Interpreter(model_path=str(TFLITE_PATH))
    interp.allocate_tensors()
    return interp


def compute_logmel(audio):
    mel = librosa.feature.melspectrogram(
        y=audio, sr=SAMPLE_RATE, n_fft=N_FFT,
        hop_length=HOP_LENGTH, n_mels=N_MELS,
        fmin=60.0, fmax=7600.0, power=2.0
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
    return probs[0], lm, latency


# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sound Alert System — ESP32-S3 Simulator",
    page_icon="🔊",
    layout="wide"
)

st.title("🔊 Sound Alert System — ESP32-S3 Simulation Dashboard")
st.markdown("*Real-time simulation of the embedded inference pipeline*")

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ ESP32-S3 Config")
    st.metric("CPU", "240 MHz")
    st.metric("Model size", "36.4 KB")
    st.metric("Parameters", "18,630")
    st.metric("Quantization", "INT8")
    st.divider()

    st.subheader("Confidence thresholds")
    thresholds = {}
    for i, cls in enumerate(CLASSES):
        thresholds[cls] = st.slider(cls, 0.0, 1.0, THRESHOLDS[i], 0.05, key=f"t_{cls}")
    st.divider()

    smooth_frames = st.slider("Temporal smoothing (frames)", 1, 7, 3)

# ── Main area ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📁 File Simulation", "📊 Results", "🏗️ Architecture", "📈 Model Metrics"])

with tab1:
    st.subheader("Upload or select audio to simulate")

    source = st.radio("Audio source:", ["Dataset samples", "Upload file"], horizontal=True)

    if source == "Dataset samples":
        raw_dir = Path("data/raw")
        available = {}
        for cls in CLASSES:
            d = raw_dir / cls
            if d.exists():
                files = [f for f in d.glob("*.*")
                         if f.suffix in {".wav", ".mp3", ".ogg"} and not f.stem.startswith("aug_")]
                if files:
                    available[cls] = files[:5]

        if available:
            col1, col2 = st.columns(2)
            with col1:
                selected_class = st.selectbox("Class", list(available.keys()))
            with col2:
                selected_file = st.selectbox("File", available[selected_class],
                                             format_func=lambda f: f.name)
            audio_path = str(selected_file)
        else:
            st.error("No dataset files found. Run preprocessing first.")
            st.stop()
    else:
        uploaded = st.file_uploader("Upload .wav / .mp3", type=["wav", "mp3"])
        if uploaded:
            tmp = Path("/tmp/sim_upload.wav")
            tmp.write_bytes(uploaded.read())
            audio_path = str(tmp)
        else:
            st.info("Upload an audio file to start simulation")
            st.stop()

    if st.button("▶ Run Simulation", type="primary", use_container_width=True):
        interp = load_model()
        audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)

        # Sliding window
        stride = SAMPLE_RATE // 2
        smooth_buf = np.ones((smooth_frames, NUM_CLASSES)) / NUM_CLASSES
        s_idx = 0
        results = []
        progress = st.progress(0)

        n_frames = max(1, (len(audio) - SAMPLE_RATE) // stride + 1)

        for i, start in enumerate(range(0, len(audio) - SAMPLE_RATE + 1, stride)):
            window = audio[start:start + SAMPLE_RATE]
            probs, lm, lat = run_inference(interp, window)

            smooth_buf[s_idx] = probs
            s_idx = (s_idx + 1) % smooth_frames
            avg = smooth_buf.mean(axis=0)

            best = int(np.argmax(avg))
            cls_name = CLASSES[best]
            conf = avg[best]
            alert = conf >= thresholds[cls_name] and best != 5

            results.append({
                "frame": i,
                "time": round(start / SAMPLE_RATE, 2),
                "class": cls_name,
                "confidence": round(float(conf), 4),
                "alert": alert,
                "latency_ms": round(lat, 2),
                "est_esp32_ms": round(lat * 8, 2),
                "probs": {c: round(float(avg[j]), 4) for j, c in enumerate(CLASSES)},
            })
            progress.progress((i + 1) / n_frames)

        st.session_state["sim_results"] = results
        st.session_state["sim_audio"] = audio
        st.session_state["sim_path"] = audio_path

        # ── Quick results ────────────────────────────────────
        alerts = [r for r in results if r["alert"]]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Frames", len(results))
        col2.metric("Alerts", len(alerts))
        col3.metric("Avg latency (ESP32)", f"{np.mean([r['est_esp32_ms'] for r in results]):.1f}ms")
        col4.metric("Duration", f"{len(audio)/SAMPLE_RATE:.1f}s")

        # LED simulation
        st.subheader("LED & Vibration Output")
        for r in results:
            if r["alert"]:
                led_color = LED_COLORS_HEX[r["class"]]
                st.markdown(
                    f'<div style="background:{led_color};color:white;padding:8px;'
                    f'border-radius:4px;margin:2px 0;font-family:monospace">'
                    f'🚨 t={r["time"]:.1f}s | {r["class"]} | conf={r["confidence"]:.3f}</div>',
                    unsafe_allow_html=True
                )

with tab2:
    if "sim_results" in st.session_state:
        results = st.session_state["sim_results"]

        # Confidence over time
        fig = go.Figure()
        for i, cls in enumerate(CLASSES):
            y_vals = [r["probs"][cls] for r in results]
            fig.add_trace(go.Scatter(
                x=[r["time"] for r in results], y=y_vals,
                name=cls, line=dict(color=COLORS[i], width=2)
            ))
            fig.add_hline(y=thresholds[cls], line_dash="dot",
                          line_color=COLORS[i], opacity=0.4)
        fig.update_layout(
            title="Smoothed confidence per class over time",
            xaxis_title="Time (s)", yaxis_title="Confidence",
            height=450, yaxis_range=[0, 1.05]
        )
        st.plotly_chart(fig, use_container_width=True)

        # Latency histogram
        fig2 = px.histogram(
            x=[r["est_esp32_ms"] for r in results],
            nbins=30, title="Estimated ESP32-S3 inference latency distribution",
            labels={"x": "Latency (ms)", "y": "Count"}
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Run a simulation in the 'File Simulation' tab first")

with tab3:
    st.subheader("System Architecture")
    st.code("""
    ┌─────────────┐     ┌──────────────┐     ┌─────────────┐
    │  INMP441     │────▶│  Log-Mel     │────▶│  DS-CNN     │
    │  Microphone  │ I2S │  64 channels │     │  INT8 TFLite│
    │  (16 kHz)    │     │  32ms window │     │  36.4 KB    │
    └─────────────┘     └──────────────┘     └──────┬──────┘
                                                     │
                         ┌──────────────┐            │
                         │  Temporal    │◀───────────┘
                         │  Smoothing   │
                         │  (3 frames)  │
                         └──────┬───────┘
                                │
                    ┌───────────┴──────────┐
                    │                      │
              ┌─────┴─────┐         ┌──────┴──────┐
              │  WS2812B  │         │  Vibration  │
              │  RGB LEDs │         │  Motor      │
              └───────────┘         └─────────────┘
    """, language=None)

    st.subheader("Knowledge Distillation Pipeline")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 🎓 Teacher")
        st.markdown("- 4-layer CNN\n- 423K params\n- ~1.7 MB\n- 60% train acc")
    with col2:
        st.markdown("### 🔄 Distillation")
        st.markdown("- Temperature: 6.0\n- α = 0.7 (KD weight)\n- 40 epochs\n- Loss: αKL + (1-α)CE")
    with col3:
        st.markdown("### 🎯 Student")
        st.markdown("- DS-CNN (4 blocks)\n- 18K params\n- 36.4 KB INT8\n- 95.4% test acc")

with tab4:
    st.subheader("Model Performance")

    # F1 per class
    f1_scores = [0.945, 1.000, 0.900, 0.933, 0.906, 0.993]
    fig = go.Figure(go.Bar(
        x=f1_scores, y=CLASSES, orientation='h',
        marker_color=COLORS, text=[f"{f:.3f}" for f in f1_scores],
        textposition='outside'
    ))
    fig.add_vline(x=0.946, line_dash="dash", annotation_text="Macro F1 = 0.946")
    fig.update_layout(title="F1 Score per class (TFLite INT8)", xaxis_range=[0.7, 1.08], height=350)
    st.plotly_chart(fig, use_container_width=True)

    # Confusion matrix image
    cm_path = Path("models/student/confusion_matrix.png")
    if cm_path.exists():
        st.image(str(cm_path), caption="Confusion Matrix (TFLite INT8 on test set)")

    # Model comparison
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Teacher vs Student")
        st.dataframe({
            "Metric": ["Parameters", "Size", "Accuracy", "Latency (ESP32)"],
            "Teacher CNN": ["423,430", "1.7 MB", "42.4%*", "~800ms"],
            "Student DS-CNN": ["18,630", "36.4 KB", "95.4%", "~2ms"],
        }, use_container_width=True)
        st.caption("*Teacher val_acc is low due to overfitting on small dataset — student generalizes better via distillation")

    with col2:
        st.markdown("### Quantization Impact")
        st.dataframe({
            "Format": ["Float32", "INT8 (QAT)"],
            "Size": ["72.8 KB", "36.4 KB"],
            "Accuracy": ["94.1%", "95.4%"],
            "Latency": ["~12ms", "~2ms"],
        }, use_container_width=True)
