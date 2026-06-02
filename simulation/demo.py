"""
SafeSound — Professional Demo v2
=================================
Pastel-themed, tabbed navigation, live mic, signal processing, IoT architecture.

Run:
    cd /Users/Apple/Desktop/iot
    source venv/bin/activate
    streamlit run simulation/demo.py
"""

import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import tensorflow as tf
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time
import soundfile as sf
import io
import threading
import queue
from pathlib import Path

try:
    import sounddevice as sd
    SD_AVAILABLE = True
except ImportError:
    SD_AVAILABLE = False

# ── Constants ─────────────────────────────────────────────────────────────────

SAMPLE_RATE = 16000
N_MELS      = 64
N_FFT       = 512
HOP_LENGTH  = 256
NUM_CLASSES = 6
CLASSES     = ["fire_alarm", "baby_cry", "choking", "car_horn", "doorbell", "background"]
TFLITE_PATH = Path("models/student/dscnn_int8.tflite")
RAW_DIR     = Path("data/raw")

THRESHOLDS = {
    "fire_alarm": 0.75, "baby_cry": 0.80, "choking": 0.75,
    "car_horn":   0.80, "doorbell": 0.90, "background": 0.50,
}
PRIORITY = {
    "fire_alarm": "CRITICAL", "choking": "CRITICAL",
    "baby_cry":   "HIGH",     "car_horn": "HIGH",
    "doorbell":   "MEDIUM",   "background": "NONE",
}
VIBRATION = {
    "fire_alarm": "5 rapid pulses", "choking": "5 rapid pulses",
    "baby_cry":   "3 pulses",       "car_horn": "3 pulses",
    "doorbell":   "1 long pulse",   "background": "None",
}
VIBRATE_CLASSES = {"fire_alarm", "choking", "baby_cry", "car_horn"}

# ── Pastel Palette ────────────────────────────────────────────────────────────

C = {
    "bg":       "#F6F8FF",
    "bg2":      "#EEF2FF",
    "card":     "#FFFFFF",
    "border":   "#E2E8F0",
    "shadow":   "rgba(0,0,0,0.06)",
    "tp":       "#1A2433",
    "ts":       "#4A5568",
    "tm":       "#94A3B8",
    "blue":     "#5B8DEF",
    "blue_l":   "#EBF2FF",
    "green":    "#52C47C",
    "green_l":  "#E8F8EF",
    "amber":    "#E8974A",
    "amber_l":  "#FFF4E8",
    "red":      "#E85A6A",
    "red_l":    "#FEEAEC",
    "purple":   "#8B7DD8",
    "purple_l": "#F0EDFF",
    "teal":     "#3DBDA8",
    "teal_l":   "#E6F9F6",
}

LED_COLORS = {
    "fire_alarm": C["red"],    "baby_cry":   C["amber"],
    "choking":    C["red"],    "car_horn":   "#E8C44A",
    "doorbell":   C["blue"],   "background": C["tm"],
}

# ── Page Config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="SafeSound — Sound Alert System",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global CSS ────────────────────────────────────────────────────────────────

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

*, *::before, *::after {{ box-sizing: border-box; }}

.stApp {{
    background: {C['bg']} !important;
    color: {C['tp']} !important;
    font-family: 'Inter', -apple-system, sans-serif;
}}

#MainMenu, footer, header {{ visibility: hidden; }}
[data-testid="collapsedControl"] {{ display: none; }}
[data-testid="stSidebar"] {{ display: none !important; }}

.block-container {{ padding-top: 0; padding-bottom: 2rem; max-width: 1280px; }}

/* ── Top Tabs ── */
.stTabs [data-baseweb="tab-list"] {{
    background: {C['card']};
    border-bottom: 2px solid {C['border']};
    padding: 0 32px;
    gap: 0;
}}
.stTabs [data-baseweb="tab"] {{
    font-size: 13px;
    font-weight: 500;
    color: {C['ts']} !important;
    padding: 14px 22px;
    border-bottom: 2px solid transparent;
    background: transparent !important;
    margin-bottom: -2px;
    font-family: 'Inter', sans-serif;
}}
.stTabs [aria-selected="true"] {{
    color: {C['blue']} !important;
    border-bottom: 2px solid {C['blue']} !important;
    font-weight: 700 !important;
}}
.stTabs [data-baseweb="tab"]:hover {{
    color: {C['blue']} !important;
    background: {C['blue_l']} !important;
}}
.stTabs [data-baseweb="tab-panel"] {{ padding-top: 0 !important; }}

/* ── Cards ── */
.card {{
    background: {C['card']};
    border: 1px solid {C['border']};
    border-radius: 12px;
    padding: 24px;
    box-shadow: 0 1px 4px {C['shadow']};
    margin-bottom: 16px;
}}
.card-sm {{
    background: {C['card']};
    border: 1px solid {C['border']};
    border-radius: 10px;
    padding: 14px 16px;
    box-shadow: 0 1px 3px {C['shadow']};
}}
.card-header {{
    font-size: 10px;
    font-weight: 700;
    color: {C['tm']};
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-bottom: 14px;
}}

/* ── Badges ── */
.badge {{
    display: inline-flex;
    align-items: center;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    white-space: nowrap;
}}
.b-critical {{ background:{C['red_l']}; color:{C['red']}; border:1px solid {C['red']}33; }}
.b-high     {{ background:{C['amber_l']}; color:{C['amber']}; border:1px solid {C['amber']}33; }}
.b-medium   {{ background:{C['blue_l']}; color:{C['blue']}; border:1px solid {C['blue']}33; }}
.b-none     {{ background:{C['border']}; color:{C['tm']}; border:1px solid {C['border']}; }}

/* ── Buttons ── */
.stButton > button {{
    background: {C['blue']} !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    padding: 8px 20px !important;
    transition: all 0.15s ease !important;
}}
.stButton > button:hover {{
    background: #4a7de0 !important;
    box-shadow: 0 4px 14px {C['blue']}44 !important;
}}
.stButton > button:disabled {{
    background: {C['border']} !important;
    color: {C['tm']} !important;
    box-shadow: none !important;
}}

/* ── Misc ── */
.stProgress > div > div > div > div {{ background: {C['blue']} !important; }}
.stSelectbox label, .stRadio label {{ color: {C['ts']} !important; font-size: 13px !important; }}
h1, h2, h3 {{ color: {C['tp']} !important; font-family: 'Inter', sans-serif !important; }}
.stRadio > div > label > div:first-child {{ border-color: {C['blue']} !important; }}
[data-baseweb="select"] {{ border-color: {C['border']} !important; }}

@keyframes blink {{ 0%,100%{{opacity:1}} 50%{{opacity:0.25}} }}
@keyframes pulse-glow {{
    from {{ box-shadow: 0 0 16px var(--led-color, #5B8DEF); }}
    to   {{ box-shadow: 0 0 32px var(--led-color, #5B8DEF), 0 0 60px var(--led-color, #5B8DEF)44; }}
}}
</style>
""", unsafe_allow_html=True)

# ── Branded Header ────────────────────────────────────────────────────────────

st.markdown(f"""
<div style="
    background: {C['card']};
    border-bottom: 1px solid {C['border']};
    padding: 0 32px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    height: 58px;
">
    <div style="display:flex;align-items:center;gap:14px;">
        <div style="
            width:36px; height:36px;
            background: linear-gradient(135deg, {C['blue']}, {C['purple']});
            border-radius: 10px;
            box-shadow: 0 4px 12px {C['blue']}44;
        "></div>
        <div>
            <div style="font-size:17px;font-weight:800;color:{C['tp']};letter-spacing:-0.3px;">SafeSound</div>
            <div style="font-size:11px;color:{C['tm']};font-weight:400;margin-top:-1px;">IoT Sound Alert Bracelet — ESP32-S3</div>
        </div>
    </div>
    <div style="display:flex;align-items:center;gap:20px;">
        <div style="display:flex;align-items:center;gap:6px;">
            <div style="width:7px;height:7px;border-radius:50%;background:{C['green']};animation:blink 2s ease infinite;"></div>
            <span style="font-size:11px;color:{C['ts']};">Model Ready · 36.4 KB</span>
        </div>
        <div style="display:flex;align-items:center;gap:6px;">
            <div style="width:7px;height:7px;border-radius:50%;background:{C['blue']};"></div>
            <span style="font-size:11px;color:{C['ts']};">TFLite INT8</span>
        </div>
        <div style="
            padding:5px 14px;
            background: {C['green_l']};
            border: 1px solid {C['green']}44;
            border-radius: 20px;
            font-size:11px; font-weight:700; color:{C['green']};
        ">95.4% Accuracy</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Model & Feature Extraction ────────────────────────────────────────────────

@st.cache_resource
def load_model():
    interp = tf.lite.Interpreter(model_path=str(TFLITE_PATH))
    interp.allocate_tensors()
    return interp


def compute_logmel(audio: np.ndarray) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=audio.astype(np.float32), sr=SAMPLE_RATE,
        n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS,
        fmin=60.0, fmax=7600.0, power=2.0,
    )
    lm = librosa.power_to_db(mel, ref=np.max)
    lm = (lm - lm.min()) / (lm.max() - lm.min() + 1e-8)
    return lm.astype(np.float32)


def run_inference(interp, audio: np.ndarray):
    inp  = interp.get_input_details()[0]
    out  = interp.get_output_details()[0]
    s_in, z_in   = inp["quantization"]
    s_out, z_out = out["quantization"]
    lm = compute_logmel(audio)
    x_q = (lm[np.newaxis, ..., np.newaxis] / s_in + z_in).astype(np.int8)
    t0 = time.perf_counter()
    interp.set_tensor(inp["index"], x_q)
    interp.invoke()
    latency = (time.perf_counter() - t0) * 1000
    y_q   = interp.get_tensor(out["index"])
    probs = (y_q.astype(np.float32) - z_out) * s_out
    return probs[0], latency

# ── Session State ─────────────────────────────────────────────────────────────

def _init():
    defs = {
        "mic_active":         False,
        "mic_queue":          queue.Queue(maxsize=5),
        "mic_error_queue":    queue.Queue(maxsize=3),
        "mic_stop_event":     threading.Event(),
        "mic_thread":         None,
        "latest_result":      None,
        "detection_history":  [],
        "frame_count":        0,
    }
    for k, v in defs.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()

# ── Microphone Worker (background thread) ─────────────────────────────────────

def _mic_worker(q: queue.Queue, stop_event: threading.Event, interp,
                error_q: queue.Queue):
    smooth_buf = np.ones((3, NUM_CLASSES)) / NUM_CLASSES
    s_idx = 0
    # Find a reliable input device — prefer MacBook mic, fall back to default
    device = None
    try:
        devs = sd.query_devices()
        for i, d in enumerate(devs):
            if d["max_input_channels"] >= 1 and "MacBook Pro Microphone" in d["name"]:
                device = i
                break
    except Exception:
        pass  # use default

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                            dtype="float32", blocksize=SAMPLE_RATE,
                            device=device) as stream:
            while not stop_event.is_set():
                audio, _ = stream.read(SAMPLE_RATE)
                audio     = audio.flatten()
                probs, lat = run_inference(interp, audio)
                smooth_buf[s_idx] = probs
                s_idx = (s_idx + 1) % 3
                avg   = smooth_buf.mean(axis=0)
                best  = int(np.argmax(avg))
                cls   = CLASSES[best]
                conf  = float(avg[best])
                alert = conf >= THRESHOLDS[cls] and cls != "background"
                result = {
                    "class":      cls,
                    "confidence": conf,
                    "probs":      avg.tolist(),
                    "alert":      alert,
                    "vibrating":  alert and cls in VIBRATE_CLASSES,
                    "latency":    lat,
                    "timestamp":  time.time(),
                    "audio":      audio.copy(),
                    "rms":        float(np.sqrt(np.mean(audio ** 2))),
                    "device":     str(device),
                }
                if q.full():
                    try: q.get_nowait()
                    except queue.Empty: pass
                q.put_nowait(result)
    except Exception as e:
        try: error_q.put_nowait(str(e))
        except queue.Full: pass

# ── Rendering Helpers ─────────────────────────────────────────────────────────

def _bracelet_html(cls_name: str, confidence: float, alert: bool, vibrating: bool) -> str:
    led   = LED_COLORS.get(cls_name, C["border"])
    glow  = f"0 0 20px {led}88, 0 0 40px {led}44" if alert else "none"
    pulse = f"pulse-glow 0.6s ease-in-out infinite alternate" if vibrating else "none"
    leds  = "".join(
        f'<div style="width:11px;height:11px;border-radius:50%;'
        f'background:{led if alert else C["border"]};'
        f'opacity:{"1" if alert else "0.3"};'
        f'box-shadow:{"0 0 10px " + led if alert else "none"};'
        f'margin:2px;transition:all 0.4s;"></div>'
        for _ in range(8)
    )
    vib_label = VIBRATION.get(cls_name, "None") if vibrating else "Idle"
    vib_color = C["amber"] if vibrating else C["tm"]

    return f"""
    <style>
        @keyframes pulse-glow {{
            from {{ box-shadow:{glow}; }}
            to   {{ box-shadow:0 0 36px {led}aa, 0 0 70px {led}55; }}
        }}
    </style>
    <div style="display:flex;flex-direction:column;align-items:center;padding:16px 0;">
        <div style="
            animation:{pulse};
            background:linear-gradient(160deg,{C['card']},{C['bg']});
            border:2px solid {led if alert else C['border']};
            border-radius:20px; padding:20px;
            width:220px;
            box-shadow:{glow};
            transition:all 0.4s ease;
        ">
            <div style="display:flex;justify-content:center;flex-wrap:wrap;margin-bottom:14px;">{leds}</div>
            <div style="text-align:center;margin-bottom:14px;">
                <div style="font-size:12px;font-weight:800;color:{led if alert else C['ts']};
                            font-family:monospace;text-transform:uppercase;letter-spacing:2px;">
                    {cls_name.replace('_', ' ')}
                </div>
                <div style="font-size:11px;color:{C['tm']};margin-top:3px;font-family:monospace;">
                    {confidence:.1%} confidence
                </div>
            </div>
            <div style="background:{C['bg']};border-radius:8px;padding:8px 10px;
                        border:1px solid {C['border']};text-align:center;">
                <div style="font-size:9px;color:{C['tm']};text-transform:uppercase;
                            letter-spacing:0.5px;margin-bottom:2px;">Motor</div>
                <div style="font-size:11px;font-weight:700;color:{vib_color};
                            font-family:monospace;">{vib_label}</div>
            </div>
        </div>
        <div style="width:22px;height:10px;background:{C['card']};
                    border:1px solid {C['border']};border-top:none;
                    border-radius:0 0 6px 6px;"></div>
    </div>
    """


def _conf_bars_html(probs: np.ndarray, best: int, label: str = "") -> str:
    hdr = f'<div class="card-header">{label + " — " if label else ""}Confidence Scores</div>'
    rows = ""
    for j, cls in enumerate(CLASSES):
        p  = float(probs[j])
        bw = int(p * 100)
        lc = LED_COLORS[cls]
        active = j == best
        tc = C["tp"] if active else C["ts"]
        fw = "700" if active else "400"
        arrow = f'<span style="color:{lc};font-size:10px;font-weight:700;">&#9650;</span>' if active else ""
        rows += f"""
        <div style="display:flex;align-items:center;gap:10px;margin:7px 0;">
            <span style="width:84px;font-size:11px;color:{tc};font-weight:{fw};
                         text-align:right;font-family:monospace;flex-shrink:0;">
                {cls.replace('_', ' ')}
            </span>
            <div style="flex:1;height:7px;background:{C['border']};border-radius:4px;overflow:hidden;">
                <div style="width:{bw}%;height:100%;background:{lc};border-radius:4px;
                            transition:width 0.3s ease;"></div>
            </div>
            <span style="width:36px;font-size:11px;color:{lc if active else C['tm']};
                         font-family:monospace;font-weight:{fw};">{p:.0%}</span>
            {arrow}
        </div>"""
    return f'<div class="card">{hdr}<div style="margin-top:4px;">{rows}</div></div>'


def _plot_wave_spec(audio: np.ndarray) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 3.8), facecolor="white")
    t = np.linspace(0, 1, len(audio))
    ax1.plot(t, audio, lw=0.5, color=C["blue"], alpha=0.9)
    ax1.fill_between(t, audio, alpha=0.1, color=C["blue"])
    ax1.axhline(0, color=C["border"], lw=0.6)
    ax1.set_xlim(0, 1)
    ax1.set_ylabel("Amplitude", fontsize=8, color=C["ts"])
    ax1.tick_params(labelsize=7, colors=C["tm"])
    ax1.set_facecolor("white")
    for s in ax1.spines.values(): s.set_edgecolor(C["border"])

    mel = librosa.feature.melspectrogram(
        y=audio.astype(np.float32), sr=SAMPLE_RATE,
        n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS,
        fmin=60.0, fmax=7600.0, power=2.0,
    )
    lm = librosa.power_to_db(mel, ref=np.max)
    librosa.display.specshow(
        lm, sr=SAMPLE_RATE, hop_length=HOP_LENGTH,
        x_axis="time", y_axis="mel", ax=ax2, cmap="Blues",
    )
    ax2.set_ylabel("Mel freq (Hz)", fontsize=8, color=C["ts"])
    ax2.set_xlabel("Time (s)", fontsize=8, color=C["ts"])
    ax2.tick_params(labelsize=7, colors=C["tm"])
    ax2.set_facecolor("white")
    for s in ax2.spines.values(): s.set_edgecolor(C["border"])
    plt.tight_layout(pad=0.6)
    return fig


# Shared CSS injected inside components.html iframes
_IFRAME_CSS = f"""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
*{{margin:0;padding:0;box-sizing:border-box;}}
body{{background:transparent;font-family:'Inter',sans-serif;color:{C['tp']};}}
.card{{background:{C['card']};border:1px solid {C['border']};border-radius:12px;padding:22px;box-shadow:0 1px 4px rgba(0,0,0,0.06);margin-bottom:12px;}}
.card-header{{font-size:10px;font-weight:700;color:{C['tm']};text-transform:uppercase;letter-spacing:1.2px;margin-bottom:12px;}}
.badge{{display:inline-flex;align-items:center;padding:3px 10px;border-radius:20px;font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.5px;white-space:nowrap;}}
.b-critical{{background:{C['red_l']};color:{C['red']};border:1px solid {C['red']}33;}}
.b-high{{background:{C['amber_l']};color:{C['amber']};border:1px solid {C['amber']}33;}}
.b-medium{{background:{C['blue_l']};color:{C['blue']};border:1px solid {C['blue']}33;}}
.b-none{{background:{C['border']};color:{C['tm']};border:1px solid {C['border']};}}
</style>
"""


def _iframe(html_body: str, height: int):
    components.html(f"<html><head>{_IFRAME_CSS}</head><body>{html_body}</body></html>",
                    height=height, scrolling=False)

# ── Live Display Fragment ─────────────────────────────────────────────────────

@st.fragment(run_every=1)
def _live_display():
    active = st.session_state.get("mic_active", False)
    latest = st.session_state.get("latest_result")

    if not active and latest is None:
        st.markdown(f"""
        <div style="
            background:{C['blue_l']};border:1px solid {C['blue']}33;
            border-radius:12px;padding:40px;text-align:center;
        ">
            <div style="font-size:14px;font-weight:600;color:{C['blue']};margin-bottom:6px;">
                Microphone is idle
            </div>
            <div style="font-size:12px;color:{C['ts']};">
                Press <strong>Start Listening</strong> to begin real-time classification.
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    # Pull from queue
    result = None
    try:
        result = st.session_state.mic_queue.get_nowait()
        st.session_state.latest_result = result
        st.session_state.frame_count   = st.session_state.frame_count + 1
        if result["alert"]:
            h = st.session_state.detection_history
            h.insert(0, {**result, "frame": st.session_state.frame_count})
            st.session_state.detection_history = h[:20]
    except queue.Empty:
        result = latest

    if result is None:
        # Check if thread reported an error
        err = None
        try: err = st.session_state.mic_error_queue.get_nowait()
        except queue.Empty: pass
        if err:
            st.session_state.mic_active = False
            st.markdown(f"""
            <div style="background:{C['red_l']};border:1px solid {C['red']}44;border-left:4px solid {C['red']};
                        border-radius:10px;padding:16px 20px;">
                <div style="font-size:12px;font-weight:700;color:{C['red']};margin-bottom:4px;">
                    Microphone Error
                </div>
                <div style="font-size:11px;color:{C['ts']};font-family:monospace;">{err}</div>
                <div style="font-size:11px;color:{C['ts']};margin-top:6px;">
                    Try: set your MacBook Pro Microphone as default input in System Preferences → Sound.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background:{C['bg2']};border-radius:10px;padding:20px;text-align:center;
                        font-size:12px;color:{C['tm']};">
                Capturing first audio window...
            </div>
            """, unsafe_allow_html=True)
        return

    probs = np.array(result["probs"])
    best  = int(np.argmax(probs))
    fc    = st.session_state.frame_count

    # Layout: bracelet left, waveform+spec right
    col_b, col_w = st.columns([1, 2])

    with col_b:
        st.markdown(
            _bracelet_html(result["class"], result["confidence"],
                           result["alert"], result["vibrating"]),
            unsafe_allow_html=True,
        )
        # Frame stats
        elapsed = fc  # 1 frame ≈ 1 second
        rms_pct = int(min(result["rms"] * 600, 100))
        st.markdown(f"""
        <div class="card-sm" style="text-align:center;margin-top:-8px;">
            <div style="font-size:10px;color:{C['tm']};font-family:monospace;">
                Frame {fc} &nbsp;|&nbsp; {result['latency']:.1f} ms host &nbsp;|&nbsp; ~{result['latency']*8:.0f} ms ESP32
            </div>
            <div style="margin-top:10px;">
                <div style="font-size:9px;color:{C['tm']};text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px;">Signal Level</div>
                <div style="height:6px;background:{C['border']};border-radius:3px;overflow:hidden;">
                    <div style="width:{rms_pct}%;height:100%;background:{C['green']};border-radius:3px;transition:width 0.3s;"></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_w:
        audio_arr = np.array(result["audio"]) if not isinstance(result["audio"], np.ndarray) else result["audio"]
        fig = _plot_wave_spec(audio_arr)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # Confidence bars (full width)
    priority = PRIORITY[result["class"]]
    badge_cls = {"CRITICAL": "b-critical", "HIGH": "b-high", "MEDIUM": "b-medium", "NONE": "b-none"}[priority]
    if result["alert"]:
        led = LED_COLORS[result["class"]]
        st.markdown(f"""
        <div style="background:{led}15;border:1px solid {led}44;border-left:4px solid {led};
                    border-radius:10px;padding:12px 18px;margin-bottom:12px;
                    display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;">
            <div>
                <span style="font-size:13px;font-weight:700;color:{led};">
                    {result['class'].replace('_', ' ').upper()} DETECTED
                </span>
                <span style="font-size:11px;color:{C['ts']};margin-left:12px;">
                    Priority: <span class="badge {badge_cls}">{priority}</span>
                </span>
            </div>
            <div style="font-size:11px;color:{C['ts']};font-family:monospace;">
                {result['confidence']:.1%} conf &nbsp;|&nbsp; {VIBRATION[result['class']]}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(_conf_bars_html(probs, best, f"Frame {fc}"), unsafe_allow_html=True)

    # Detection history
    hist = st.session_state.detection_history
    if hist:
        rows = ""
        for h in hist[:10]:
            ts    = time.strftime("%H:%M:%S", time.localtime(h["timestamp"]))
            color = LED_COLORS.get(h["class"], C["tm"])
            pri   = PRIORITY[h["class"]]
            bc    = {"CRITICAL": "b-critical", "HIGH": "b-high", "MEDIUM": "b-medium", "NONE": "b-none"}[pri]
            rows += f"""
            <div style="display:flex;align-items:center;gap:12px;padding:9px 0;
                        border-bottom:1px solid {C['border']};">
                <div style="width:10px;height:10px;border-radius:50%;background:{color};flex-shrink:0;"></div>
                <span style="font-family:monospace;font-size:11px;color:{C['tm']};flex-shrink:0;">{ts}</span>
                <span style="font-size:12px;font-weight:600;color:{C['tp']};flex:1;">
                    {h['class'].replace('_', ' ').title()}
                </span>
                <span style="font-family:monospace;font-size:11px;color:{color};">{h['confidence']:.0%}</span>
                <span class="badge {bc}">{pri}</span>
            </div>
            """
        st.markdown(f"""
        <div class="card">
            <div class="card-header">Alert History ({len(hist)} events)</div>
            {rows}
        </div>
        """, unsafe_allow_html=True)
    elif fc > 2:
        st.markdown(f"""
        <div class="card-sm" style="text-align:center;color:{C['tm']};font-size:12px;padding:20px;">
            No alerts detected — only background noise or speech in {fc} frames.
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════

(tab_overview, tab_live, tab_signal, tab_ml, tab_iot) = st.tabs([
    "Overview",
    "Live Detection",
    "Signal Processing",
    "ML Architecture",
    "IoT System",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

with tab_overview:
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {C['blue_l']} 0%, {C['purple_l']} 100%);
        border: 1px solid {C['border']};
        border-radius: 16px;
        padding: 52px 48px;
        text-align: center;
        margin: 28px 0 32px 0;
    ">
        <div style="
            display:inline-block; width:60px; height:60px;
            background: linear-gradient(135deg, {C['blue']}, {C['purple']});
            border-radius: 18px;
            box-shadow: 0 8px 24px {C['blue']}44;
            margin-bottom: 20px;
        "></div>
        <h1 style="font-size:42px;font-weight:800;color:{C['tp']};margin:0;letter-spacing:-0.5px;">
            SafeSound
        </h1>
        <p style="font-size:16px;color:{C['ts']};margin:14px auto 0;max-width:560px;line-height:1.65;">
            A wearable IoT bracelet that classifies critical ambient sounds in real-time
            and alerts deaf or hard-of-hearing users through haptic and visual feedback —
            entirely on-device, no cloud required.
        </p>
        <div style="display:flex;justify-content:center;gap:10px;margin-top:22px;flex-wrap:wrap;">
            <span class="badge b-critical">Zero Cloud Dependency</span>
            <span class="badge b-medium">On-Device ML · TFLite INT8</span>
            <span class="badge b-none">Sub-2ms Inference</span>
            <span class="badge b-high">Knowledge Distillation · 23x</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Key metrics
    cols = st.columns(4)
    metrics = [
        ("Model Size",      "36.4 KB", "INT8 quantized TFLite",         C["blue"],   C["blue_l"]),
        ("Test Accuracy",   "95.4%",   "On clean test set (n=153)",      C["green"],  C["green_l"]),
        ("Inference",       "< 2 ms",  "On ESP32-S3 @ 240 MHz",          C["purple"], C["purple_l"]),
        ("Sound Classes",   "6",       "5 alert types + 1 background",   C["amber"],  C["amber_l"]),
    ]
    for col, (label, val, sub, color, bg) in zip(cols, metrics):
        with col:
            st.markdown(f"""
            <div style="background:{bg};border:1px solid {color}33;border-radius:12px;
                        padding:22px;border-left:3px solid {color};margin-bottom:0;">
                <div style="font-size:10px;font-weight:700;color:{color};text-transform:uppercase;
                            letter-spacing:1px;margin-bottom:8px;">{label}</div>
                <div style="font-size:30px;font-weight:800;color:{C['tp']};line-height:1;">{val}</div>
                <div style="font-size:11px;color:{C['ts']};margin-top:7px;">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # How it works — 4-step pipeline
    step_cols = st.columns(4)
    steps = [
        ("Capture",   f"INMP441 I2S microphone at 16 kHz. Continuous DMA streaming with no buffering lag.",                     C["blue"]),
        ("Extract",   "Log-Mel spectrogram — 64 Mel channels, 32 ms window, 16 ms hop. Output: 64×63 tensor.",                 C["purple"]),
        ("Classify",  "DS-CNN INT8 (18 K params, 36.4 KB). Knowledge-distilled from a 423 K ResNet-style teacher.",            C["teal"]),
        ("Alert",     "WS2812B LED color + ERM motor vibration. Per-class thresholds. Temporal 3-frame majority vote.",         C["amber"]),
    ]
    for i, (col, (title, desc, color)) in enumerate(zip(step_cols, steps)):
        with col:
            arrow = f'<div style="position:absolute;top:50%;right:-13px;width:26px;height:2px;background:{C["border"]};"></div>' if i < 3 else ""
            st.markdown(f"""
            <div style="position:relative;background:{C['card']};border:1px solid {C['border']};
                        border-radius:12px;padding:22px;text-align:center;
                        box-shadow:0 1px 4px {C['shadow']};min-height:165px;">
                {arrow}
                <div style="
                    width:40px;height:40px;border-radius:50%;
                    background:{color}1a;border:2px solid {color}44;
                    display:flex;align-items:center;justify-content:center;
                    margin:0 auto 14px auto;
                ">
                    <span style="font-size:16px;font-weight:800;color:{color};">{i+1}</span>
                </div>
                <div style="font-size:13px;font-weight:700;color:{C['tp']};margin-bottom:7px;">{title}</div>
                <div style="font-size:11px;color:{C['ts']};line-height:1.55;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Sound classes table
    grid = "display:grid;grid-template-columns:1.2fr 0.8fr 1fr 1.1fr 0.55fr;gap:0;align-items:center;"
    hdr  = f"font-size:10px;color:{C['tm']};text-transform:uppercase;letter-spacing:1px;font-weight:700;"
    rb   = f"border-bottom:1px solid {C['border']};"
    tbl  = f"""
    <div class="card">
        <div class="card-header">Detected Sound Classes</div>
        <div style="{grid} padding:10px 0;{rb}">
            <span style="{hdr}">Class</span>
            <span style="{hdr}">Priority</span>
            <span style="{hdr}">LED Color</span>
            <span style="{hdr}">Vibration Pattern</span>
            <span style="{hdr}">Threshold</span>
        </div>
    """
    for cls in CLASSES:
        p   = PRIORITY[cls]
        bc  = {"CRITICAL":"b-critical","HIGH":"b-high","MEDIUM":"b-medium","NONE":"b-none"}[p]
        dot = LED_COLORS[cls]
        tbl += f"""
        <div style="{grid} padding:12px 0;{rb}">
            <span style="font-size:13px;font-weight:600;color:{C['tp']};">{cls.replace('_',' ').title()}</span>
            <span><span class="badge {bc}">{p}</span></span>
            <span style="display:flex;align-items:center;gap:8px;">
                <span style="width:12px;height:12px;border-radius:50%;background:{dot};
                             box-shadow:0 0 6px {dot}88;display:inline-block;"></span>
                <span style="font-size:11px;color:{C['ts']};font-family:monospace;">{dot}</span>
            </span>
            <span style="font-size:12px;color:{C['ts']};">{VIBRATION[cls]}</span>
            <span style="font-size:12px;color:{C['ts']};font-family:monospace;">{THRESHOLDS[cls]:.0%}</span>
        </div>"""
    tbl += "</div>"
    _iframe(tbl, 390)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — LIVE DETECTION
# ══════════════════════════════════════════════════════════════════════════════

with tab_live:
    if not TFLITE_PATH.exists():
        st.error(f"Model not found: {TFLITE_PATH}")
        st.stop()

    interp = load_model()

    st.markdown(f"""
    <div style="padding:22px 0 14px 0;">
        <h2 style="font-size:22px;font-weight:800;margin-bottom:4px;">Live Detection</h2>
        <p style="color:{C['ts']};font-size:13px;margin:0;">
            Real-time TFLite inference — observe how the bracelet responds to each sound.
        </p>
    </div>
    """, unsafe_allow_html=True)

    mode = st.radio("Input mode", ["Live Microphone", "Audio File"],
                    horizontal=True, label_visibility="collapsed")

    st.markdown(f'<div style="border-top:1px solid {C["border"]};margin:8px 0 20px 0;"></div>',
                unsafe_allow_html=True)

    # ── LIVE MIC ─────────────────────────────────────────────────────────────
    if mode == "Live Microphone":
        if not SD_AVAILABLE:
            st.warning("sounddevice is not installed. Run: pip install sounddevice")
        else:
            st.markdown(f"""
            <div style="background:{C['blue_l']};border:1px solid {C['blue']}33;border-radius:12px;
                        padding:14px 20px;margin-bottom:18px;">
                <div style="font-size:12px;font-weight:700;color:{C['blue']};margin-bottom:3px;
                            text-transform:uppercase;letter-spacing:0.5px;">Live Microphone Mode</div>
                <div style="font-size:12px;color:{C['ts']};">
                    Captures 1-second windows continuously. Alerts only for fire alarm, baby cry,
                    choking, and car horn. Background and speech are silently ignored.
                    Temporal smoothing: 3-frame majority vote.
                </div>
            </div>
            """, unsafe_allow_html=True)

            btn1, btn2, btn3 = st.columns([1, 1, 3])
            with btn1:
                start_btn = st.button("Start Listening", type="primary",
                                      use_container_width=True,
                                      disabled=st.session_state.mic_active)
            with btn2:
                stop_btn = st.button("Stop", use_container_width=True,
                                     disabled=not st.session_state.mic_active)
            with btn3:
                if st.session_state.mic_active:
                    st.markdown(f"""
                    <div style="display:flex;align-items:center;gap:8px;padding:8px 14px;
                                background:{C['green_l']};border-radius:8px;
                                border:1px solid {C['green']}44;height:38px;margin-top:1px;">
                        <div style="width:8px;height:8px;border-radius:50%;
                                    background:{C['green']};animation:blink 1s ease infinite;"></div>
                        <span style="font-size:12px;font-weight:700;color:{C['green']};">Listening</span>
                        <span style="font-size:11px;color:{C['ts']};">
                            — Frame {st.session_state.frame_count}
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="display:flex;align-items:center;gap:8px;padding:8px 14px;
                                background:{C['border']};border-radius:8px;height:38px;margin-top:1px;">
                        <div style="width:8px;height:8px;border-radius:50%;background:{C['tm']};"></div>
                        <span style="font-size:12px;color:{C['ts']};">Microphone idle</span>
                    </div>
                    """, unsafe_allow_html=True)

            if start_btn and not st.session_state.mic_active:
                st.session_state.mic_stop_event.clear()
                st.session_state.frame_count        = 0
                st.session_state.detection_history  = []
                st.session_state.latest_result      = None
                for _q in (st.session_state.mic_queue, st.session_state.mic_error_queue):
                    while not _q.empty():
                        try: _q.get_nowait()
                        except queue.Empty: break
                t = threading.Thread(
                    target=_mic_worker,
                    args=(st.session_state.mic_queue,
                          st.session_state.mic_stop_event,
                          interp,
                          st.session_state.mic_error_queue),
                    daemon=True,
                )
                t.start()
                st.session_state.mic_thread = t
                st.session_state.mic_active = True
                st.rerun()

            if stop_btn and st.session_state.mic_active:
                st.session_state.mic_stop_event.set()
                st.session_state.mic_active = False
                st.rerun()

            st.markdown("<br>", unsafe_allow_html=True)
            _live_display()

    # ── AUDIO FILE ────────────────────────────────────────────────────────────
    else:
        col_ctrl, col_brace = st.columns([2.5, 1.5])

        available = {}
        for cls in CLASSES:
            d = RAW_DIR / cls
            if d.exists():
                files = sorted([f for f in d.glob("*.*")
                                 if f.suffix in {".wav",".mp3",".ogg"}
                                 and not f.stem.startswith("aug_")])
                if files:
                    available[cls] = files[:10]

        with col_ctrl:
            src = st.radio("Source", ["Dataset samples", "Upload file"], horizontal=True)
            audio_path = None

            if src == "Dataset samples" and available:
                c1, c2 = st.columns(2)
                with c1:
                    sel_cls = st.selectbox("Class", list(available.keys()),
                                           format_func=lambda c: c.replace("_"," ").title())
                with c2:
                    sel_file = st.selectbox("File", available.get(sel_cls, []),
                                            format_func=lambda f: f.name)
                audio_path = str(sel_file) if sel_cls in available else None
            elif src == "Upload file":
                up = st.file_uploader("Drop .wav / .mp3 / .ogg", type=["wav","mp3","ogg"],
                                      label_visibility="collapsed")
                if up:
                    tmp = Path("/tmp/ss_upload.wav")
                    tmp.write_bytes(up.read())
                    audio_path = str(tmp)

            run_btn = st.button("Run Inference", type="primary",
                                use_container_width=True, disabled=audio_path is None)

        with col_brace:
            brace_slot = st.empty()
            brace_slot.markdown(_bracelet_html("background", 0.0, False, False),
                                unsafe_allow_html=True)

        prog_slot  = st.empty()
        conf_slot  = st.empty()
        wave_slot  = st.empty()
        tl_slot    = st.empty()
        res_slot   = st.empty()

        if run_btn and audio_path:
            audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
            stride   = SAMPLE_RATE // 2
            n_frames = max(1, (len(audio) - SAMPLE_RATE) // stride + 1)

            buf = io.BytesIO()
            sf.write(buf, audio, SAMPLE_RATE, format="WAV")
            buf.seek(0)
            with col_ctrl:
                st.audio(buf, format="audio/wav")

            smooth_buf = np.ones((3, NUM_CLASSES)) / NUM_CLASSES
            s_idx, all_events, latencies = 0, [], []
            prog = prog_slot.progress(0, text="Processing...")

            for i, start in enumerate(range(0, len(audio) - SAMPLE_RATE + 1, stride)):
                window = audio[start:start + SAMPLE_RATE]
                probs, lat = run_inference(interp, window)
                latencies.append(lat)
                smooth_buf[s_idx] = probs
                s_idx = (s_idx + 1) % 3
                avg   = smooth_buf.mean(axis=0)
                best  = int(np.argmax(avg))
                cls   = CLASSES[best]
                conf  = float(avg[best])
                alert = conf >= THRESHOLDS[cls] and cls != "background"
                vib   = alert and cls in VIBRATE_CLASSES
                all_events.append({"frame":i+1,"time":i*0.5,"class":cls,
                                   "confidence":conf,"alert":alert,"vibrating":vib})

                brace_slot.markdown(_bracelet_html(cls, conf, alert, vib), unsafe_allow_html=True)
                conf_slot.markdown(_conf_bars_html(avg, best, f"Frame {i+1}/{n_frames}"),
                                   unsafe_allow_html=True)

                fig = _plot_wave_spec(window)
                wave_slot.pyplot(fig, use_container_width=True)
                plt.close(fig)

                prog.progress((i+1)/n_frames, text=f"Frame {i+1}/{n_frames} — {cls}")
                time.sleep(0.15)

            prog.empty()

            # Timeline
            tl_html = f"""
            <div class="card">
                <div class="card-header">Detection Timeline — {n_frames} frames</div>
                <div style="display:flex;gap:3px;margin-top:10px;align-items:flex-end;height:48px;">
            """
            for e in all_events:
                color = LED_COLORS[e["class"]] if e["alert"] else C["border"]
                h     = max(4, int(e["confidence"] * 48))
                tl_html += f'<div style="flex:1;height:{h}px;background:{color};border-radius:2px;min-width:4px;" title="{e["class"]} {e["confidence"]:.0%}"></div>'
            tl_html += "</div></div>"
            _iframe(tl_html, 105)

            alert_events = [e for e in all_events if e["alert"]]
            if alert_events:
                f  = alert_events[-1]
                lc = LED_COLORS[f["class"]]
                pr = PRIORITY[f["class"]]
                bc = {"CRITICAL":"b-critical","HIGH":"b-high","MEDIUM":"b-medium","NONE":"b-none"}[pr]
                _iframe(f"""
                <div class="card" style="border-left:4px solid {lc};">
                    <div style="display:flex;align-items:center;gap:16px;">
                        <div style="width:44px;height:44px;border-radius:50%;
                                    background:{lc}1a;border:2px solid {lc};
                                    display:flex;align-items:center;justify-content:center;flex-shrink:0;">
                            <div style="width:16px;height:16px;border-radius:50%;background:{lc};"></div>
                        </div>
                        <div>
                            <div style="font-size:17px;font-weight:800;color:{C['tp']};">
                                {f['class'].replace('_',' ').upper()} DETECTED
                            </div>
                            <div style="font-size:12px;color:{C['ts']};margin-top:3px;">
                                {len(alert_events)} alert frames / {n_frames} total &nbsp;|&nbsp;
                                Avg latency: {np.mean(latencies)*8:.1f} ms (ESP32 est.) &nbsp;|&nbsp;
                                Priority: <span class="badge {bc}">{pr}</span>
                            </div>
                        </div>
                    </div>
                </div>
                """, 110)
            else:
                _iframe(f"""
                <div class="card">
                    <div style="display:flex;align-items:center;gap:16px;">
                        <div style="width:44px;height:44px;border-radius:50%;background:{C['border']};
                                    display:flex;align-items:center;justify-content:center;">
                            <div style="width:16px;height:16px;border-radius:50%;background:{C['tm']};"></div>
                        </div>
                        <div>
                            <div style="font-size:17px;font-weight:800;color:{C['tp']};">
                                No Alert — Background Noise Only
                            </div>
                            <div style="font-size:12px;color:{C['ts']};margin-top:3px;">
                                {n_frames} frames processed — no threshold exceeded
                            </div>
                        </div>
                    </div>
                </div>
                """, 110)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — SIGNAL PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

with tab_signal:
    st.markdown(f"""
    <div style="padding:22px 0 14px 0;">
        <h2 style="font-size:22px;font-weight:800;margin-bottom:4px;">Signal Processing Pipeline</h2>
        <p style="color:{C['ts']};font-size:13px;margin:0;">
            Step-by-step visualization from raw microphone samples to the INT8 tensor fed to the model.
        </p>
    </div>
    """, unsafe_allow_html=True)

    sp_avail = {}
    for cls in CLASSES:
        d = RAW_DIR / cls
        if d.exists():
            files = sorted([f for f in d.glob("*.*")
                             if f.suffix in {".wav",".mp3",".ogg"}
                             and not f.stem.startswith("aug_")])
            if files:
                sp_avail[cls] = files[:6]

    c1, c2 = st.columns(2)
    with c1:
        sp_cls = st.selectbox("Sound class", list(sp_avail.keys()),
                               format_func=lambda c: c.replace("_"," ").title(), key="sp_cls")
    with c2:
        sp_file = st.selectbox("Sample", sp_avail.get(sp_cls, []),
                                format_func=lambda f: f.name, key="sp_file")

    def _step_banner(n, label, detail, color):
        return f"""
        <div style="background:{color}1a;border-left:3px solid {color};border-radius:0 8px 8px 0;
                    padding:12px 18px;margin:20px 0 10px 0;display:flex;align-items:center;gap:12px;">
            <div style="width:26px;height:26px;border-radius:50%;background:{color}22;
                        border:2px solid {color}44;display:flex;align-items:center;justify-content:center;flex-shrink:0;">
                <span style="font-size:12px;font-weight:800;color:{color};">{n}</span>
            </div>
            <div>
                <span style="font-size:12px;font-weight:700;color:{color};text-transform:uppercase;
                             letter-spacing:0.4px;">{label}</span>
                <span style="font-size:11px;color:{C['ts']};margin-left:10px;">{detail}</span>
            </div>
        </div>"""

    def _stat_cards(*items):
        cols = st.columns(len(items))
        for col, (label, val) in zip(cols, items):
            with col:
                st.markdown(f"""
                <div class="card-sm" style="margin-bottom:0;">
                    <div style="font-size:10px;color:{C['tm']};font-weight:700;text-transform:uppercase;
                                letter-spacing:0.5px;margin-bottom:4px;">{label}</div>
                    <div style="font-size:17px;font-weight:800;color:{C['tp']};
                                font-family:monospace;">{val}</div>
                </div>
                """, unsafe_allow_html=True)

    if sp_file and Path(str(sp_file)).exists():
        audio_sp, _ = librosa.load(str(sp_file), sr=SAMPLE_RATE, mono=True, duration=1.0)
        if len(audio_sp) < SAMPLE_RATE:
            audio_sp = np.pad(audio_sp, (0, SAMPLE_RATE - len(audio_sp)))

        # ── Step 1: Raw Waveform ──────────────────────────────────────────────
        st.markdown(_step_banner(1, "Raw Waveform", "16,000 samples at 16 kHz = 1 second window",
                                  C["blue"]), unsafe_allow_html=True)
        fig1, ax1 = plt.subplots(figsize=(10, 1.9), facecolor="white")
        t = np.linspace(0, 1, len(audio_sp))
        ax1.plot(t, audio_sp, lw=0.5, color=C["blue"], alpha=0.9)
        ax1.fill_between(t, audio_sp, alpha=0.1, color=C["blue"])
        ax1.axhline(0, color=C["border"], lw=0.6)
        ax1.set_xlim(0, 1)
        ax1.set_ylabel("Amplitude", fontsize=8, color=C["ts"])
        ax1.set_xlabel("Time (s)", fontsize=8, color=C["ts"])
        ax1.tick_params(labelsize=7, colors=C["tm"])
        ax1.set_facecolor("white")
        for s in ax1.spines.values(): s.set_edgecolor(C["border"])
        plt.tight_layout(pad=0.4)
        st.pyplot(fig1, use_container_width=True)
        plt.close(fig1)
        _stat_cards(("Sample Rate", "16,000 Hz"), ("Samples", "16,000"),
                    ("Duration", "1.000 s"), ("RMS Energy", f"{np.sqrt(np.mean(audio_sp**2)):.4f}"))

        # ── Step 2: Mel Spectrogram ───────────────────────────────────────────
        st.markdown(_step_banner(2, "Mel Spectrogram",
                                  "STFT (N_FFT=512, hop=256 → 32 ms / 16 ms) mapped to 64 Mel bins",
                                  C["purple"]), unsafe_allow_html=True)
        mel = librosa.feature.melspectrogram(
            y=audio_sp.astype(np.float32), sr=SAMPLE_RATE,
            n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS,
            fmin=60.0, fmax=7600.0, power=2.0,
        )
        lm_raw = librosa.power_to_db(mel, ref=np.max)
        fig2, ax2 = plt.subplots(figsize=(10, 2.3), facecolor="white")
        img = librosa.display.specshow(lm_raw, sr=SAMPLE_RATE, hop_length=HOP_LENGTH,
                                       x_axis="time", y_axis="mel", ax=ax2,
                                       cmap="Purples", fmin=60, fmax=7600)
        cb = plt.colorbar(img, ax=ax2, format="%+2.0f dB", pad=0.01)
        cb.ax.tick_params(labelsize=7, colors=C["tm"])
        ax2.set_ylabel("Mel freq (Hz)", fontsize=8, color=C["ts"])
        ax2.set_xlabel("Time (s)", fontsize=8, color=C["ts"])
        ax2.tick_params(labelsize=7, colors=C["tm"])
        ax2.set_facecolor("white")
        for s in ax2.spines.values(): s.set_edgecolor(C["border"])
        plt.tight_layout(pad=0.4)
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)
        _stat_cards(("Output Shape", f"64 × {mel.shape[1]}"), ("Mel Bins", "64"),
                    ("Freq Range", "60 – 7600 Hz"), ("FFT Window", "32 ms"))

        # ── Step 3: Log-Mel Normalized ────────────────────────────────────────
        lm_norm = (lm_raw - lm_raw.min()) / (lm_raw.max() - lm_raw.min() + 1e-8)
        st.markdown(_step_banner(3, "Normalized Log-Mel",
                                  "Min-max normalization to [0, 1] — this is the float32 model input",
                                  C["teal"]), unsafe_allow_html=True)
        fig3, ax3 = plt.subplots(figsize=(10, 2.3), facecolor="white")
        im3 = ax3.imshow(lm_norm, aspect="auto", origin="lower",
                          cmap="GnBu", extent=[0, 1, 60, 7600])
        cb3 = plt.colorbar(im3, ax=ax3, pad=0.01)
        cb3.ax.tick_params(labelsize=7, colors=C["tm"])
        ax3.set_ylabel("Freq (Hz)", fontsize=8, color=C["ts"])
        ax3.set_xlabel("Time (s)", fontsize=8, color=C["ts"])
        ax3.tick_params(labelsize=7, colors=C["tm"])
        ax3.set_facecolor("white")
        for s in ax3.spines.values(): s.set_edgecolor(C["border"])
        plt.tight_layout(pad=0.4)
        st.pyplot(fig3, use_container_width=True)
        plt.close(fig3)
        _stat_cards(("Value Range", "[0.000 – 1.000]"), ("Data Type", "float32"),
                    ("Shape", f"(1, 64, {mel.shape[1]}, 1)"),
                    ("Memory", f"{lm_norm.nbytes / 1024:.1f} KB"))

        # ── Step 4: INT8 Quantization ─────────────────────────────────────────
        st.markdown(_step_banner(4, "INT8 Quantization",
                                  "Affine quantization: x_int8 = x_float / scale + zero_point. 4× memory reduction.",
                                  C["amber"]), unsafe_allow_html=True)
        if TFLITE_PATH.exists():
            _interp = load_model()
            inp_d   = _interp.get_input_details()[0]
            s_in, z_in = inp_d["quantization"]
            x_q = (lm_norm[np.newaxis, ..., np.newaxis] / s_in + z_in).astype(np.int8)

            fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(10, 2.4), facecolor="white")
            ax4a.imshow(lm_norm, aspect="auto", origin="lower", cmap="Blues")
            ax4a.set_title(f"Float32  range [{lm_norm.min():.2f}, {lm_norm.max():.2f}]",
                           fontsize=9, color=C["ts"], pad=6)
            ax4a.tick_params(labelsize=7, colors=C["tm"])
            ax4a.set_facecolor("white")
            for s in ax4a.spines.values(): s.set_edgecolor(C["border"])

            ax4b.imshow(x_q[0, :, :, 0], aspect="auto", origin="lower", cmap="Oranges")
            ax4b.set_title(f"INT8  range [{x_q.min()}, {x_q.max()}]  scale={s_in:.5f}  zp={z_in}",
                           fontsize=9, color=C["ts"], pad=6)
            ax4b.tick_params(labelsize=7, colors=C["tm"])
            ax4b.set_facecolor("white")
            for s in ax4b.spines.values(): s.set_edgecolor(C["border"])
            plt.tight_layout(pad=0.4)
            st.pyplot(fig4, use_container_width=True)
            plt.close(fig4)

            _stat_cards(
                ("Scale Factor", f"{s_in:.6f}"),
                ("Zero Point",   str(z_in)),
                ("Float32 size", f"{lm_norm.nbytes / 1024:.1f} KB"),
                ("INT8 size",    f"{x_q.nbytes / 1024:.1f} KB — 4× smaller"),
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ML ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════

with tab_ml:
    st.markdown(f"""
    <div style="padding:22px 0 14px 0;">
        <h2 style="font-size:22px;font-weight:800;margin-bottom:4px;">ML Architecture</h2>
        <p style="color:{C['ts']};font-size:13px;margin:0;">
            Knowledge distillation pipeline, quantization-aware training, and per-class performance.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Distillation banner
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,{C['blue_l']},{C['purple_l']});
                border:1px solid {C['blue']}33;border-radius:14px;
                padding:22px 28px;margin-bottom:24px;">
        <div style="display:flex;align-items:flex-start;justify-content:space-between;flex-wrap:wrap;gap:16px;">
            <div style="max-width:420px;">
                <div style="font-size:14px;font-weight:800;color:{C['tp']};margin-bottom:6px;">
                    Knowledge Distillation
                </div>
                <div style="font-size:12px;color:{C['ts']};line-height:1.6;">
                    The large teacher CNN (423 K params) is trained first on the full dataset.
                    Its soft probability outputs at temperature T=6 carry <em>dark knowledge</em>
                    about inter-class similarity. The student DS-CNN learns from both hard labels
                    (weight α=0.7) and these soft labels simultaneously, achieving 95.4% accuracy
                    in just 18 K parameters — beating the teacher's 94.1%.
                </div>
            </div>
            <div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap;">
                <div style="text-align:center;padding:14px 18px;background:{C['card']};
                            border-radius:10px;border:1px solid {C['border']};">
                    <div style="font-size:22px;font-weight:800;color:{C['tp']};">423 K</div>
                    <div style="font-size:10px;color:{C['tm']};text-transform:uppercase;letter-spacing:0.5px;margin-top:2px;">Teacher</div>
                </div>
                <div style="font-size:22px;color:{C['tm']};">&#8594;</div>
                <div style="text-align:center;padding:14px 18px;background:{C['card']};
                            border-radius:10px;border:1px solid {C['blue']}44;">
                    <div style="font-size:22px;font-weight:800;color:{C['blue']};">18 K</div>
                    <div style="font-size:10px;color:{C['tm']};text-transform:uppercase;letter-spacing:0.5px;margin-top:2px;">Student</div>
                </div>
                <div style="font-size:22px;color:{C['tm']};">&#8594;</div>
                <div style="text-align:center;padding:14px 18px;background:{C['green_l']};
                            border-radius:10px;border:1px solid {C['green']}44;">
                    <div style="font-size:22px;font-weight:800;color:{C['green']};">36.4 KB</div>
                    <div style="font-size:10px;color:{C['tm']};text-transform:uppercase;letter-spacing:0.5px;margin-top:2px;">INT8 TFLite</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Teacher vs Student
    col_t, col_s = st.columns(2)

    def _row(label, val, color=None):
        vc = color or C["tp"]
        return (f'<div style="display:flex;justify-content:space-between;padding:10px 0;'
                f'border-bottom:1px solid {C["border"]};">'
                f'<span style="font-size:12px;color:{C["ts"]};">{label}</span>'
                f'<span style="font-size:12px;font-weight:700;color:{vc};font-family:monospace;">{val}</span>'
                f'</div>')

    with col_t:
        st.markdown(
            f'<div class="card"><div class="card-header">Teacher Model — Training Only</div>'
            + _row("Architecture", "ResNet-style CNN")
            + _row("Parameters", "423,174")
            + _row("Model size", "~1.7 MB")
            + _row("Accuracy", "94.1%", C["green"])
            + _row("Deployable on ESP32", "No — exceeds SRAM", C["red"])
            + "</div>",
            unsafe_allow_html=True,
        )

    with col_s:
        st.markdown(
            f'<div class="card" style="border-left:3px solid {C["blue"]};">'
            f'<div class="card-header" style="color:{C["blue"]};">Student Model — Deployed on ESP32-S3</div>'
            + _row("Architecture", "DS-CNN (3 DW-Sep blocks)")
            + _row("Parameters", "18,086")
            + _row("Model size (INT8)", "36.4 KB", C["blue"])
            + _row("Accuracy", "95.4%", C["green"])
            + _row("Deployable on ESP32", "Yes — 9.4% of SRAM", C["green"])
            + "</div>",
            unsafe_allow_html=True,
        )

    # F1 bar chart
    results = [
        ("Fire Alarm",  0.921, 0.972, 0.945, C["red"]),
        ("Baby Cry",    1.000, 1.000, 1.000, C["amber"]),
        ("Choking",     0.857, 0.947, 0.900, C["red"]),
        ("Car Horn",    0.895, 0.971, 0.933, "#E8C44A"),
        ("Doorbell",    0.958, 0.856, 0.906, C["blue"]),
        ("Background",  1.000, 0.986, 0.993, C["tm"]),
    ]

    fig_f1, ax_f1 = plt.subplots(figsize=(8, 3.6), facecolor="white")
    names  = [r[0] for r in results]
    f1s    = [r[3] for r in results]
    colors = [r[4] for r in results]
    bars   = ax_f1.barh(names, f1s, color=colors, height=0.52, alpha=0.82)
    ax_f1.set_xlim(0.80, 1.04)
    ax_f1.axvline(0.946, color=C["purple"], lw=1.3, ls="--", alpha=0.8, label="Macro avg F1 = 0.946")
    for bar, val in zip(bars, f1s):
        ax_f1.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                   f"{val:.3f}", va="center", fontsize=9, color=C["tp"], fontweight="700")
    ax_f1.set_xlabel("F1 Score", fontsize=9, color=C["ts"])
    ax_f1.legend(fontsize=8, framealpha=0)
    ax_f1.tick_params(labelsize=9, colors=C["ts"])
    ax_f1.set_facecolor("white")
    ax_f1.set_title("Per-Class F1 Score — TFLite INT8 Student", fontsize=10, color=C["tp"], pad=10)
    for s in ax_f1.spines.values(): s.set_edgecolor(C["border"])
    plt.tight_layout(pad=0.6)
    st.pyplot(fig_f1, use_container_width=True)
    plt.close(fig_f1)

    # Confusion matrix
    cm_path = Path("models/student/confusion_matrix.png")
    if cm_path.exists():
        st.markdown(f'<div class="card-header" style="margin-top:12px;">Confusion Matrix</div>',
                    unsafe_allow_html=True)
        st.image(str(cm_path), use_container_width=True)

    # Training config
    cfg_items = [
        ("Distillation temperature (T)", "6"),
        ("Alpha (α) — hard/soft loss mix", "0.7"),
        ("QAT epochs", "10"),
        ("QAT learning rate", "1 × 10⁻⁵"),
        ("Input shape (INT8)", "(1, 64, 63, 1)"),
        ("Temporal smoothing", "3-frame circular buffer · majority vote"),
    ]
    rows = "".join(
        f'<div style="padding:10px 16px;border-bottom:1px solid {C["border"]};">'
        f'<span style="font-size:11px;color:{C["ts"]};">{label}</span>'
        f'<span style="float:right;font-size:12px;font-weight:700;color:{C["tp"]};font-family:monospace;">{val}</span>'
        f'</div>'
        for label, val in cfg_items
    )
    st.markdown(
        f'<div class="card" style="margin-top:16px;"><div class="card-header">Training Configuration</div>{rows}</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — IoT SYSTEM
# ══════════════════════════════════════════════════════════════════════════════

with tab_iot:
    st.markdown(f"""
    <div style="padding:22px 0 14px 0;">
        <h2 style="font-size:22px;font-weight:800;margin-bottom:4px;">IoT System Architecture</h2>
        <p style="color:{C['ts']};font-size:13px;margin:0;">
            Hardware design, communication stack, MQTT schema, and embedded system integration.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Architecture flow diagram
    _arch_nodes = [
        ("INMP441",    "I2S Microphone",  "16 kHz",       C["blue"],   "I2S DMA"),
        ("ESP32-S3",   "DS-CNN INT8",     "240 MHz",      C["blue"],   "BLE 5.0"),
        ("Smartphone", "MQTT Bridge",     "BLE → MQTT",   C["purple"], "MQTT"),
        ("Mosquitto",  "MQTT Broker",     "port 1883",    C["teal"],   "Subscribe"),
        ("Node-RED",   "Flow Engine",     "port 1880",    C["teal"],   "Push"),
        ("Caregiver",  "Alert App",       "Real-time",    C["green"],  None),
    ]
    _arch_parts = []
    for i, (name, sub, detail, color, proto) in enumerate(_arch_nodes):
        _arch_parts.append(
            f'<div style="background:{C["card"]};border:1px solid {C["border"]};border-radius:12px;'
            f'padding:14px 16px;text-align:center;min-width:110px;'
            f'box-shadow:0 1px 4px rgba(0,0,0,0.05);">'
            f'<div style="font-size:12px;font-weight:700;color:{C["tp"]};">{name}</div>'
            f'<div style="font-size:10px;color:{C["tm"]};margin-top:2px;">{sub}</div>'
            f'<div style="font-size:10px;font-weight:600;color:{color};margin-top:4px;">{detail}</div>'
            f'</div>'
        )
        if proto:
            _arch_parts.append(
                f'<div style="display:flex;flex-direction:column;align-items:center;flex-shrink:0;">'
                f'<div style="height:2px;width:36px;background:{C["border"]};"></div>'
                f'<div style="font-size:9px;font-weight:700;color:{C["blue"]};margin-top:3px;'
                f'text-transform:uppercase;letter-spacing:0.3px;white-space:nowrap;">{proto}</div>'
                f'</div>'
            )
    _iframe(f"""
    <div class="card">
        <div class="card-header">End-to-End Communication Stack</div>
        <div style="display:flex;align-items:center;justify-content:center;
                    gap:6px;padding:10px 0;overflow-x:auto;">
            {"".join(_arch_parts)}
        </div>
    </div>
    """, 170)

    col_hw, col_iot = st.columns(2)

    with col_hw:
        # BOM
        bom = [
            ("ESP32-S3 DevKitC", "MCU",        "Dual-core LX7 @ 240 MHz · 8 MB PSRAM"),
            ("INMP441",          "Microphone",  "I2S MEMS · 16 kHz · SNR 61 dB"),
            ("WS2812B × 8",      "LED Strip",   "RGB · color-coded per class"),
            ("ERM 3V DC",        "Vibration",   "NPN transistor driver · haptic"),
            ("LiPo 3.7V 1000mAh","Battery",     "8–12 h runtime · USB-C charging"),
            ("TP4056",           "Charger IC",  "Li-Ion management + protection"),
        ]
        bom_rows = "".join(
            f'<div style="display:grid;grid-template-columns:1.2fr 0.8fr 1.6fr;gap:8px;'
            f'padding:10px 0;border-bottom:1px solid {C["border"]};">'
            f'<span style="font-size:12px;font-weight:700;color:{C["tp"]};font-family:monospace;">{part}</span>'
            f'<span style="font-size:11px;color:{C["blue"]};font-weight:600;">{comp}</span>'
            f'<span style="font-size:11px;color:{C["ts"]};">{role}</span>'
            f'</div>'
            for part, comp, role in bom
        )
        st.markdown(
            f'<div class="card"><div class="card-header">Bill of Materials</div>{bom_rows}</div>',
            unsafe_allow_html=True,
        )

        # Pins
        st.markdown(f"""
        <div class="card">
            <div class="card-header">ESP32-S3 Pin Assignments</div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:4px;">
                <div>
                    <div style="font-size:10px;color:{C['blue']};font-weight:700;text-transform:uppercase;
                                letter-spacing:0.5px;margin-bottom:7px;">I2S — INMP441</div>
                    <div style="background:{C['bg']};border:1px solid {C['border']};border-radius:8px;
                                padding:11px 14px;font-family:monospace;font-size:11px;color:{C['ts']};line-height:2;">
                        SCK &nbsp;= GPIO 14<br>WS &nbsp;&nbsp;= GPIO 15<br>SD &nbsp;&nbsp;= GPIO 13
                    </div>
                </div>
                <div>
                    <div style="font-size:10px;color:{C['purple']};font-weight:700;text-transform:uppercase;
                                letter-spacing:0.5px;margin-bottom:7px;">Outputs</div>
                    <div style="background:{C['bg']};border:1px solid {C['border']};border-radius:8px;
                                padding:11px 14px;font-family:monospace;font-size:11px;color:{C['ts']};line-height:2;">
                        LED_DATA = GPIO 5<br>MOTOR &nbsp;&nbsp;= GPIO 6<br>STATUS &nbsp;= GPIO 7
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_iot:
        # MQTT topics
        mqtt_topics = [
            ("safesound/classification", "ESP32 → Broker",    "class + confidence JSON payload", C["blue"]),
            ("safesound/alert",          "ESP32 → Broker",    "triggered when threshold exceeded", C["red"]),
            ("safesound/device/status",  "ESP32 → Broker",    "battery %, uptime, RSSI",           C["green"]),
            ("safesound/notification",   "Broker → Caregiver","push notification payload",          C["amber"]),
            ("safesound/config/thresholds","Broker → ESP32",  "remote threshold configuration",     C["purple"]),
        ]
        mqtt_rows = "".join(
            f'<div style="padding:10px 0;border-bottom:1px solid {C["border"]};">'
            f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:3px;">'
            f'<div style="width:8px;height:8px;border-radius:2px;background:{color};flex-shrink:0;"></div>'
            f'<span style="font-size:11px;font-weight:700;color:{C["tp"]};font-family:monospace;">{topic}</span>'
            f'</div>'
            f'<div style="padding-left:16px;">'
            f'<span style="font-size:10px;font-weight:700;color:{color};text-transform:uppercase;letter-spacing:0.3px;">{direction}</span>'
            f'<span style="font-size:10px;color:{C["tm"]};margin-left:8px;">{desc}</span>'
            f'</div></div>'
            for topic, direction, desc, color in mqtt_topics
        )
        st.markdown(
            f'<div class="card"><div class="card-header">MQTT Topic Schema</div>{mqtt_rows}</div>',
            unsafe_allow_html=True,
        )

        # Memory budget
        st.markdown(f"""
        <div class="card">
            <div class="card-header">ESP32-S3 Memory Budget</div>
            <div style="margin-top:4px;">
                <div style="display:flex;justify-content:space-between;margin-bottom:5px;">
                    <span style="font-size:12px;color:{C['ts']};">TFLite Model</span>
                    <span style="font-size:12px;font-weight:700;color:{C['tp']};font-family:monospace;">36.4 KB / 512 KB</span>
                </div>
                <div style="height:8px;background:{C['border']};border-radius:4px;overflow:hidden;margin-bottom:14px;">
                    <div style="width:7.1%;height:100%;background:{C['blue']};border-radius:4px;"></div>
                </div>
                <div style="display:flex;justify-content:space-between;margin-bottom:5px;">
                    <span style="font-size:12px;color:{C['ts']};">Runtime (model + buffers)</span>
                    <span style="font-size:12px;font-weight:700;color:{C['tp']};font-family:monospace;">48.3 KB / 512 KB</span>
                </div>
                <div style="height:8px;background:{C['border']};border-radius:4px;overflow:hidden;margin-bottom:10px;">
                    <div style="width:9.4%;height:100%;background:{C['green']};border-radius:4px;"></div>
                </div>
                <div style="font-size:11px;color:{C['tm']};">
                    9.4% of SRAM used — 90.6% headroom for additional features
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Power budget
    power_items = [
        ("ESP32-S3 (active, 240 MHz)", "~80 mA", C["blue"]),
        ("INMP441 microphone",         "~1.4 mA", C["purple"]),
        ("WS2812B LEDs (idle)",        "~3 mA",   C["amber"]),
        ("WS2812B LEDs (alert, 8 LEDs)","~160 mA", C["red"]),
        ("ERM vibration motor",        "~80 mA",  C["teal"]),
        ("Total (typical, no alert)",  "~90 mA",  C["green"]),
    ]
    pw_rows = "".join(
        f'<div style="display:flex;justify-content:space-between;align-items:center;'
        f'padding:9px 0;border-bottom:1px solid {C["border"]};">'
        f'<span style="font-size:12px;color:{C["ts"]};">{label}</span>'
        f'<span style="font-size:12px;font-weight:700;color:{color};font-family:monospace;">{val}</span>'
        f'</div>'
        for label, val, color in power_items
    )
    st.markdown(
        f'<div class="card" style="margin-top:8px;">'
        f'<div class="card-header">Power Budget — 3.7 V LiPo 1000 mAh → 8–12 h runtime</div>'
        f'{pw_rows}</div>',
        unsafe_allow_html=True,
    )

    # Circuit schematic
    schematic = Path("notebooks/figures/circuit_schematic.png")
    if schematic.exists():
        st.markdown(f'<div class="card-header" style="margin-top:12px;">Circuit Schematic</div>',
                    unsafe_allow_html=True)
        st.image(str(schematic), use_container_width=True)
