# import streamlit as st
# import numpy as np
# from scipy.io import wavfile
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import StandardScaler
# import os
# import joblib
# import tempfile
# import matplotlib.pyplot as plt

# # ================= PAGE CONFIG =================
# st.set_page_config(
#     page_title="Noise Level Analyzer",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# st.title("🔊 Noise Level Analyzer")
# st.markdown("Analyze audio files and classify noise levels using Machine Learning")

# # ================= FEATURE EXTRACTION =================
# def extract_features(audio_data, sample_rate):
#     audio = audio_data.astype(np.float32)

#     # Normalize PCM
#     if np.issubdtype(audio_data.dtype, np.integer):
#         audio /= np.iinfo(audio_data.dtype).max

#     # Stereo → Mono
#     if audio.ndim > 1:
#         audio = np.mean(audio, axis=1)

#     rms = np.sqrt(np.mean(audio ** 2))
#     zcr = np.sum(np.abs(np.diff(np.sign(audio)))) / (2 * len(audio))

#     return [rms, zcr], audio, sample_rate


# def calculate_db(rms):
#     return max(0, int(20 * np.log10(rms + 1e-6) + 60))


# # ================= MODEL =================
# @st.cache_resource
# def load_model():
#     model_path = "saved_model.joblib"

#     if os.path.exists(model_path):
#         data = joblib.load(model_path)
#         return data["model"], data["scaler"], "Loaded trained model"

#     # ---- Synthetic Training ----
#     X, y = [], []
#     sr = 22050
#     t = np.linspace(0, 2, int(sr * 2), endpoint=False)

#     def gen(freq, amp):
#         return amp * np.sin(2 * np.pi * freq * t)

#     classes = [
#         (0, [0.01, 0.02]),   # LOW
#         (1, [0.05, 0.08]),   # MEDIUM
#         (2, [0.15, 0.25])    # HIGH
#     ]

#     for label, amps in classes:
#         for amp in amps:
#             audio = gen(440, amp)
#             features, _, _ = extract_features(audio, sr)
#             X.append(features)
#             y.append(label)

#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)

#     model = RandomForestClassifier(
#         n_estimators=50,
#         random_state=42
#     )
#     model.fit(X_scaled, y)

#     joblib.dump({"model": model, "scaler": scaler}, model_path)
#     return model, scaler, "Trained on synthetic audio data"


# def classify_audio(model, scaler, features):
#     features_scaled = scaler.transform([features])
#     pred = model.predict(features_scaled)[0]
#     prob = model.predict_proba(features_scaled)[0][pred] * 100

#     labels = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}
#     return labels[pred], prob


# # ================= SIDEBAR =================
# st.sidebar.header("📋 Model Info")
# model, scaler, status = load_model()
# st.sidebar.success(status)

# st.sidebar.markdown("---")
# st.sidebar.info("""
# **LOW**: Quiet environment  
# **MEDIUM**: Normal speech  
# **HIGH**: Loud noise  
# """)

# # ================= MAIN UI =================
# col1, col2 = st.columns(2)

# with col1:
#     uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])

#     if uploaded_file:
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
#             tmp.write(uploaded_file.read())
#             tmp_path = tmp.name

#         sr, audio = wavfile.read(tmp_path)
#         features, audio_mono, _ = extract_features(audio, sr)

#         rms, zcr = features
#         db_value = calculate_db(rms)

#         label, confidence = classify_audio(model, scaler, features)

#         os.unlink(tmp_path)

#         with col2:
#             st.subheader("📊 Results")

#             st.audio(uploaded_file)

#             color = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}
#             st.markdown(f"## {color[label]} {label}")

#             c1, c2, c3 = st.columns(3)
#             c1.metric("Sound Level", f"{db_value} dB")
#             c2.metric("RMS", f"{rms:.4f}")
#             c3.metric("Confidence", f"{confidence:.2f}%")

#         # ===== Waveform =====
#         st.markdown("---")
#         st.subheader("🌊 Waveform")

#         fig, ax = plt.subplots(figsize=(12, 3))
#         time = np.linspace(0, len(audio_mono) / sr, len(audio_mono))
#         ax.plot(time, audio_mono)
#         ax.set_xlabel("Time (s)")
#         ax.set_ylabel("Amplitude")
#         st.pyplot(fig)

#         # ===== Precautions =====
#         st.markdown("---")
#         st.subheader("⚠️ Safety Recommendations")

#         if label == "LOW":
#             st.success("Safe environment. No precautions needed.")
#         elif label == "MEDIUM":
#             st.warning("Limit long exposure. Take breaks.")
#         else:
#             st.error("High noise! Use hearing protection.")

#     else:
#         st.info("Upload a WAV file to begin analysis")

# # ================= FOOTER =================
# st.markdown("""
# <div style="text-align:center;color:gray;font-size:12px">
# Noise Level Analyzer | ML-based RMS & ZCR Classification
# </div>
# """, unsafe_allow_html=True)





import streamlit as st
import numpy as np
from scipy.io import wavfile
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import os
import joblib
import tempfile
import matplotlib.pyplot as plt

# ========= GLOBALS =========
CLASS_LABELS = {
    0: "Traffic",
    1: "Machinery",
    2: "Speech",
    3: "Music",
    4: "Wind",
}

# Track class counts for dashboard
if "class_counts" not in st.session_state:
    st.session_state["class_counts"] = {name: 0 for name in CLASS_LABELS.values()}

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Noise Environment Classifier",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🔊 Noise Environment Classifier")
st.markdown(
    "Classify audio into **Traffic, Machinery, Speech, Music, Wind** with a live dashboard."
)


# ================= FEATURE EXTRACTION =================
def extract_features(audio_data, sample_rate):
    audio = audio_data.astype(np.float32)

    # Normalize integer PCM to [-1, 1]
    if np.issubdtype(audio_data.dtype, np.integer):
        max_val = float(np.iinfo(audio_data.dtype).max)
        if max_val > 0:
            audio /= max_val

    # Stereo → mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # Time-domain features
    rms = float(np.sqrt(np.mean(audio ** 2)) + 1e-8)
    zcr = float(np.sum(np.abs(np.diff(np.sign(audio)))) / (2 * len(audio) + 1e-8))

    # Simple spectral features
    spectrum = np.fft.rfft(audio)
    mag = np.abs(spectrum)
    freqs = np.fft.rfftfreq(len(audio), d=1.0 / sample_rate)

    if mag.sum() > 0:
        centroid = float(np.sum(freqs * mag) / (mag.sum() + 1e-8))
        spread = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * mag) / (mag.sum() + 1e-8)))
    else:
        centroid = 0.0
        spread = 0.0

    features = [rms, zcr, centroid, spread]
    return features, audio, sample_rate


def calculate_db(rms):
    return max(0, int(20 * np.log10(rms + 1e-8) + 60))


def get_noise_level(db):
    """Determine noise level (LOW/MEDIUM/HIGH) based on dB value"""
    if db < 20:
        return "LOW"
    elif db >= 21 and db <= 50:
        return "MEDIUM"
    else:  # db > 51
        return "HIGH"


# ================= MODEL =================
@st.cache_resource
def load_model():
    # New path so we don't clash with any older 3-class models
    model_path = "saved_env_model.joblib"

    if os.path.exists(model_path):
        data = joblib.load(model_path)
        return data["model"], data["scaler"], "✅ Loaded trained environment model"

    X, y = [], []
    sr = 22050
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # Synthetic generators for each environment
    def gen_traffic():
        # Low rumble + random honk-like bursts
        base = 0.3 * np.sin(2 * np.pi * 60 * t) + 0.2 * np.sin(2 * np.pi * 120 * t)
        noise = 0.2 * np.random.normal(0, 1, len(t))
        return (base + noise).astype(np.float32)

    def gen_machinery():
        # Strong steady tones (motors) + some noise
        base = (
            0.4 * np.sin(2 * np.pi * 200 * t)
            + 0.3 * np.sin(2 * np.pi * 400 * t)
            + 0.2 * np.sin(2 * np.pi * 800 * t)
        )
        noise = 0.1 * np.random.normal(0, 1, len(t))
        return (base + noise).astype(np.float32)

    def gen_speech():
        # Amplitude-modulated mid-frequency tone + noise (very rough speech-like)
        carrier = np.sin(2 * np.pi * 200 * t)
        envelope = 0.5 * (1 + np.sin(2 * np.pi * 3 * t))  # syllable-like modulation
        sig = 0.4 * envelope * carrier + 0.1 * np.random.normal(0, 1, len(t))
        return sig.astype(np.float32)

    def gen_music():
        # Mix of musical notes (C major chord-ish)
        c4 = np.sin(2 * np.pi * 261.63 * t)
        e4 = np.sin(2 * np.pi * 329.63 * t)
        g4 = np.sin(2 * np.pi * 392.00 * t)
        base = 0.3 * (c4 + e4 + g4)
        noise = 0.05 * np.random.normal(0, 1, len(t))
        return (base + noise).astype(np.float32)

    def gen_wind():
        # Broadband noise, slightly filtered to higher frequencies
        noise = np.random.normal(0, 1, len(t))
        # Simple high-pass: subtract a smoothed version
        smooth = np.convolve(noise, np.ones(500) / 500, mode="same")
        sig = 0.4 * (noise - 0.7 * smooth)
        return sig.astype(np.float32)

    generators = {
        0: gen_traffic,
        1: gen_machinery,
        2: gen_speech,
        3: gen_music,
        4: gen_wind,
    }

    # Build synthetic dataset
    samples_per_class = 40
    for label_idx, gen_fn in generators.items():
        for _ in range(samples_per_class):
            audio = gen_fn()
            # slight random gain variation
            audio = (audio * np.random.uniform(0.6, 1.2)).astype(np.float32)
            feats, _, _ = extract_features(audio, sr)
            X.append(feats)
            y.append(label_idx)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(n_estimators=80, random_state=42)
    model.fit(X_scaled, y)

    joblib.dump({"model": model, "scaler": scaler}, model_path)
    return model, scaler, "✅ Trained synthetic environment model (Traffic/Machinery/Speech/Music/Wind)"


def classify_audio(model, scaler, features):
    features_scaled = scaler.transform([features])
    probs = model.predict_proba(features_scaled)[0]
    pred_idx = int(np.argmax(probs))
    label = CLASS_LABELS[pred_idx]
    return label, probs


# ================= SIDEBAR / DASHBOARD =================
st.sidebar.header("📋 Model Info")
model, scaler, status = load_model()
st.sidebar.success(status)

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Classes")
st.sidebar.markdown(
    """
- **Traffic**: Road / vehicle noise  
- **Machinery**: Engines, industrial sounds  
- **Speech**: Human talking  
- **Music**: Songs, instruments  
- **Wind**: Wind / whooshing noise  
"""
)

st.sidebar.markdown("---")
st.sidebar.subheader("📈 Session Overview")
total_seen = sum(st.session_state["class_counts"].values())
st.sidebar.metric("Files Analyzed", total_seen)
if total_seen > 0:
    most_common = max(st.session_state["class_counts"], key=st.session_state["class_counts"].get)
    st.sidebar.metric("Most Detected", most_common)


# ================= MAIN UI =================
col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        sr, audio = wavfile.read(tmp_path)
        features, audio_mono, _ = extract_features(audio, sr)

        rms, zcr, centroid, spread = features
        db_value = calculate_db(rms)

        label, probs = classify_audio(model, scaler, features)
        os.unlink(tmp_path)

        # Update session stats
        st.session_state["class_counts"][label] += 1

        with col2:
            st.subheader("📊 Classification Result")

            st.audio(uploaded_file)

            icons = {
                "Traffic": "🚗",
                "Machinery": "⚙️",
                "Speech": "🗣️",
                "Music": "🎵",
                "Wind": "🌬️",
            }
            st.markdown(f"## {icons.get(label, '🔊')} **{label}**")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Sound Level", f"{db_value} dB")
            c2.metric("RMS", f"{rms:.4f}")
            c3.metric("ZCR", f"{zcr:.4f}")
            c4.metric("Spectral Centroid", f"{centroid:.0f} Hz")

        # ===== Noise Level Section (LOW/MEDIUM/HIGH) =====
        st.markdown("---")
        noise_level = get_noise_level(db_value)
        st.subheader("🔊 Noise Level")
        
        level_colors = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}
        level_colors_hex = {"LOW": "#00AA00", "MEDIUM": "#FFAA00", "HIGH": "#FF0000"}
        
        col_level1, col_level2, col_level3 = st.columns(3)
        
        with col_level1:
            if noise_level == "LOW":
                st.markdown(f"### {level_colors['LOW']} **LOW**")
                st.success("Quiet environment - Safe for extended exposure")
            else:
                st.markdown(f"### {level_colors['LOW']} LOW")
        
        with col_level2:
            if noise_level == "MEDIUM":
                st.markdown(f"### {level_colors['MEDIUM']} **MEDIUM**")
                st.warning("Moderate noise - Take breaks periodically")
            else:
                st.markdown(f"### {level_colors['MEDIUM']} MEDIUM")
        
        with col_level3:
            if noise_level == "HIGH":
                st.markdown(f"### {level_colors['HIGH']} **HIGH**")
                st.error("Loud noise - Use hearing protection")
            else:
                st.markdown(f"### {level_colors['HIGH']} HIGH")
        
        # Visual bar chart for noise level
        fig_level, ax_level = plt.subplots(figsize=(8, 2))
        levels = ["LOW", "MEDIUM", "HIGH"]
        values = [1 if l == noise_level else 0 for l in levels]
        colors_level = [level_colors_hex[l] for l in levels]
        bars = ax_level.bar(levels, values, color=colors_level, alpha=0.7, edgecolor="black", linewidth=2)
        ax_level.set_ylabel("Detected Level", fontsize=10)
        ax_level.set_ylim(0, 1.2)
        ax_level.set_title(f"Current Noise Level: {noise_level}", fontsize=12, fontweight="bold")
        st.pyplot(fig_level, use_container_width=True)

        # ===== Dynamic Precautions Section =====
        st.markdown("---")
        st.subheader("⚠️ Precautions & Safety Recommendations")
        
        if noise_level == "LOW":
            st.success(f"""
            ✅ **Low Noise Level ({db_value} dB) - Safe Environment**
            
            - ✅ Safe for extended exposure (8+ hours)
            - ✅ Ideal for studying, office work, and concentration
            - ✅ No hearing protection required
            - ✅ Suitable for libraries, homes, quiet offices, and classrooms
            - ✅ No health risks associated with this noise level
            """)
        elif noise_level == "MEDIUM":
            st.warning(f"""
            ⚠️ **Medium Noise Level ({db_value} dB) - Moderate Exposure**
            
            - ⚠️ Prolonged exposure may cause fatigue and mild discomfort
            - ⚠️ Take regular breaks every 1-2 hours in quieter areas
            - ⚠️ Avoid using high-volume headphones or earbuds
            - ⚠️ Recommended maximum exposure: 8 hours per day
            - ⚠️ Suitable for streets, busy offices, and normal conversations
            - 💡 Consider noise-cancelling headphones if working in this environment long-term
            """)
        else:  # HIGH
            st.error(f"""
            🔴 **High Noise Level ({db_value} dB) - DANGEROUS - Immediate Action Required**
            
            - 🚨 **RISK OF HEARING DAMAGE** with prolonged exposure
            - 🚨 **USE HEARING PROTECTION IMMEDIATELY** (earplugs or noise-cancelling headphones)
            - 🚨 **Limit exposure to less than 1 hour per day** maximum
            - 🚨 **Take frequent breaks** in quiet environments (every 15-30 minutes)
            - 🚨 **Avoid continuous listening** - give your ears time to recover
            - 🚨 Common sources: traffic, construction sites, loud machinery, concerts, industrial areas
            - 🚨 **Consult health guidelines** for workplace safety compliance
            - 💡 If exposed regularly, consider professional hearing protection and regular hearing tests
            """)
        
        # Additional context based on sound type
        st.markdown("---")
        st.subheader("📋 Additional Context")
        
        context_messages = {
            "Traffic": "🚗 Traffic noise can be particularly harmful due to constant exposure. Use noise barriers or ear protection when near busy roads.",
            "Machinery": "⚙️ Machinery noise often contains harmful frequencies. Always use appropriate PPE in industrial settings.",
            "Speech": "🗣️ Speech at high volumes can still cause hearing damage. Maintain safe distances from loudspeakers.",
            "Music": "🎵 Music enjoyment shouldn't come at the cost of hearing. Follow the 60/60 rule: 60% volume for 60 minutes max.",
            "Wind": "🌬️ Wind noise is generally less harmful but can still cause discomfort. Protect your ears in windy conditions."
        }
        
        st.info(context_messages.get(label, "Monitor your exposure and take appropriate precautions based on the noise level."))

        # ===== Probability bar chart (dynamic dashboard element) =====
        st.markdown("---")
        st.subheader("📉 Class Probabilities")

        fig_prob, ax_prob = plt.subplots(figsize=(8, 3))
        class_names = [CLASS_LABELS[i] for i in range(len(CLASS_LABELS))]
        colors = ["#ff7f0e", "#9467bd", "#2ca02c", "#1f77b4", "#17becf"]
        bars = ax_prob.bar(class_names, probs * 100, color=colors, alpha=0.8)
        ax_prob.set_ylabel("Probability (%)")
        ax_prob.set_ylim(0, 100)
        for bar in bars:
            height = bar.get_height()
            ax_prob.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 1,
                f"{height:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        st.pyplot(fig_prob, use_container_width=True)

        # ===== Waveform =====
        st.markdown("---")
        st.subheader("🌊 Audio Waveform")

        fig_wav, ax_wav = plt.subplots(figsize=(12, 3))
        time = np.linspace(0, len(audio_mono) / sr, len(audio_mono))
        ax_wav.plot(time, audio_mono, linewidth=0.6, color="steelblue")
        ax_wav.set_xlabel("Time (s)")
        ax_wav.set_ylabel("Amplitude")
        ax_wav.grid(alpha=0.3)
        st.pyplot(fig_wav, use_container_width=True)

        # ===== Dynamic session summary chart =====
        st.markdown("---")
        st.subheader("📊 Session Class Distribution")
        counts = st.session_state["class_counts"]
        labels_list = list(counts.keys())
        values_list = list(counts.values())

        fig_sess, ax_sess = plt.subplots(figsize=(8, 3))
        ax_sess.bar(labels_list, values_list, color=colors)
        ax_sess.set_ylabel("Count")
        st.pyplot(fig_sess, use_container_width=True)

    else:
        st.info("Upload a WAV file to start analysis")

# ================= FOOTER =================
st.markdown(
    """
<div style="text-align:center;color:gray;font-size:12px">
Noise Environment Classifier | Traffic • Machinery • Speech • Music • Wind
</div>
""",
    unsafe_allow_html=True,
)
