"""
🇹🇳 ASR Tunisien — Interface de démonstration
Projet de Fin d'Études | Whisper Fine-Tuned pour le dialecte tunisien
"""

import os
import time
import tempfile
import io

import numpy as np
import streamlit as st
import librosa
import soundfile as sf
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from streamlit_mic_recorder import mic_recorder


import os
from pydub import AudioSegment

ffmpeg_path = r"C:\Users\Bilel\Downloads\ffmpeg-8.1-essentials_build\ffmpeg-8.1-essentials_build\bin\ffmpeg.exe"

# On force Python à utiliser ce chemin pour FFmpeg
AudioSegment.converter = ffmpeg_path
os.environ["PATH"] += os.pathsep + os.path.dirname(ffmpeg_path)


import gdown
import os

# IDs de tes dossiers Drive (à récupérer via le lien de partage)
# Exemple : si ton lien est https://drive.google.com/drive/folders/1abc... l'ID est 1abc...
ID_V1 = "12YdiqXixSf9fDSVUXu4viysFwXu2hh3O"
ID_V2 = "1CNH-YHd1iwwABofGchZAiteRXVqIwLyU"

def download_models():
    if not os.path.exists("whisper-v1"):
        gdown.download_folder(id=ID_V1, output="whisper-v1", quiet=False)
    if not os.path.exists("whisper-v2"):
        gdown.download_folder(id=ID_V2, output="whisper-v2", quiet=False)

# Appelle la fonction avant de charger les modèles
download_models()

# ─────────────────────────────────────────────
#  CONFIGURATION GLOBALE
# ─────────────────────────────────────────────
MODEL_V1_PATH = "whisper-v1/whisper-small-tunisian-FINAL"
MODEL_V2_PATH = "whisper-v2/whisper-small-tunisian-V2-ROBUST"

SAMPLE_RATE   = 16_000   # Whisper attend 16 kHz
CHUNK_SECONDS = 28       # Fenêtre < 30 s pour éviter les OOM
LANGUAGE      = "ar"     # Arabe (tunisien)
TASK          = "transcribe"

st.set_page_config(
    page_title="ASR Tunisien 🇹🇳",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CSS PERSONNALISÉ
# ─────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Palette ── */
    :root {
        --primary:   #E63946;
        --secondary: #1D3557;
        --accent:    #F4A261;
        --bg:        #F8F9FA;
        --card:      #FFFFFF;
        --muted:     #6C757D;
        --border:    #DEE2E6;
        --success:   #2DC653;
        --radius:    12px;
    }

    /* ── Fond global ── */
    .stApp { background-color: var(--bg); }

    /* ── Hero banner ── */
    .hero {
        background: linear-gradient(135deg, var(--secondary) 0%, #457B9D 100%);
        border-radius: var(--radius);
        padding: 2rem 2.5rem;
        color: #fff;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(29,53,87,.25);
    }
    .hero h1 { font-size: 2.2rem; margin: 0 0 .4rem; font-weight: 800; }
    .hero p  { font-size: 1.05rem; margin: 0; opacity: .88; }

    /* ── Carte résultat ── */
    .result-card {
        background: var(--card);
        border-radius: var(--radius);
        border: 1px solid var(--border);
        padding: 1.4rem 1.6rem;
        margin-top: .8rem;
        box-shadow: 0 2px 8px rgba(0,0,0,.06);
    }
    .result-card h3 { margin: 0 0 .6rem; font-size: 1.05rem; color: var(--secondary); }
    .transcription {
        font-size: 1.15rem;
        line-height: 1.7;
        color: #212529;
        font-family: 'Noto Naskh Arabic', 'Amiri', serif;
        direction: rtl;
        text-align: right;
        background: #F1F5FB;
        border-radius: 8px;
        padding: .9rem 1.1rem;
        margin-top: .5rem;
    }

    /* ── Badge métrique ── */
    .metric-badge {
        display: inline-flex;
        align-items: center;
        gap: .35rem;
        background: var(--secondary);
        color: #fff;
        border-radius: 999px;
        padding: .25rem .8rem;
        font-size: .82rem;
        font-weight: 600;
        margin-top: .7rem;
    }
    .badge-fast  { background: var(--success); }
    .badge-slow  { background: var(--accent); color: #333; }

    /* ── Chip technologie (sidebar) ── */
    .tech-chip {
        display: inline-block;
        background: #EDF2FF;
        color: var(--secondary);
        border-radius: 999px;
        padding: .2rem .75rem;
        font-size: .78rem;
        font-weight: 600;
        margin: .2rem .15rem;
        border: 1px solid #C5D3F6;
    }

    /* ── Séparateur section ── */
    .section-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: var(--secondary);
        border-left: 4px solid var(--primary);
        padding-left: .7rem;
        margin: 1.5rem 0 .8rem;
    }

    /* ── Bouton principal ── */
    div[data-testid="stButton"] > button {
        background: linear-gradient(135deg, var(--primary), #C1121F);
        color: #fff !important;
        border: none;
        border-radius: 8px;
        padding: .55rem 1.6rem;
        font-weight: 700;
        font-size: 1rem;
        transition: opacity .2s, transform .1s;
        box-shadow: 0 3px 10px rgba(230,57,70,.35);
    }
    div[data-testid="stButton"] > button:hover { opacity: .9; transform: translateY(-1px); }

    /* ── Masquer le filigrane Streamlit ── */
    #MainMenu, footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
#  CHARGEMENT DES MODÈLES (mis en cache)
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    """Charge le processeur et le modèle Whisper fine-tuné."""
    processor = WhisperProcessor.from_pretrained(model_path)
    model     = WhisperForConditionalGeneration.from_pretrained(model_path)
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    model     = model.to(device)
    model.eval()
    return processor, model, device


# ─────────────────────────────────────────────
#  PRÉTRAITEMENT AUDIO
# ─────────────────────────────────────────────

def preprocess_audio(audio_bytes: bytes) -> np.ndarray:
    """
    Charge les bytes audio, convertit en mono 16 kHz via Librosa.
    Retourne un tableau float32 normalisé.
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        waveform, sr = librosa.load(tmp_path, sr=SAMPLE_RATE, mono=True)
    finally:
        os.unlink(tmp_path)

    # Normalisation peak
    peak = np.max(np.abs(waveform))
    if peak > 0:
        waveform = waveform / peak

    return waveform.astype(np.float32)


def chunk_audio(waveform: np.ndarray, chunk_sec: int = CHUNK_SECONDS) -> list[np.ndarray]:
    """Découpe l'audio en segments de `chunk_sec` secondes."""
    chunk_len = chunk_sec * SAMPLE_RATE
    return [waveform[i : i + chunk_len] for i in range(0, len(waveform), chunk_len)]


# ─────────────────────────────────────────────
#  INFÉRENCE
# ─────────────────────────────────────────────

def transcribe(waveform: np.ndarray, processor, model, device: str) -> tuple[str, float]:
    """
    Transcrit un tableau audio (float32, 16 kHz).
    Gère automatiquement le découpage si > 30 s.
    Retourne (texte, temps_inférence_secondes).
    """
    chunks = chunk_audio(waveform)
    texts  = []

    forced_ids = processor.get_decoder_prompt_ids(language=LANGUAGE, task=TASK)

    t0 = time.perf_counter()

    for chunk in chunks:
        inputs = processor(
            chunk,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
        ).input_features.to(device)

        with torch.no_grad():
            predicted_ids = model.generate(
                inputs,
                forced_decoder_ids=forced_ids,
            )

        text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
        texts.append(text)

    elapsed = time.perf_counter() - t0
    return " ".join(texts), elapsed


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        """
        <div style='text-align:center; margin-bottom:1rem;'>
            <span style='font-size:3rem;'>🇹🇳</span>
            <h2 style='margin:.3rem 0 0; color:#1D3557; font-size:1.25rem;'>ASR Tunisien</h2>
            <p style='color:#6C757D; font-size:.85rem; margin:.2rem 0 0;'>Projet de Fin d'Études</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 📖 À propos du projet")
    st.markdown(
        """
        Ce projet explore la reconnaissance automatique de la parole (**ASR**) pour 
        le **dialecte tunisien** (Darija), une langue à faibles ressources.

        Deux modèles **Whisper-Small** fine-tunés sont comparés :

        | Modèle | Description |
        |--------|-------------|
        | **V1 — Baseline** | Fine-tuning standard |
        | **V2 — Robust** | + augmentation (Speed, Bruit) |

        Les modèles sont évalués sur le **WER** (Word Error Rate).
        """
    )

    st.markdown("### 🛠️ Technologies")
    techs = [
        "OpenAI Whisper", "HuggingFace 🤗", "PyTorch",
        "Librosa", "TEDxTN Corpus", "Streamlit",
        "Python 3.11", "CUDA / CPU",
    ]
    chips = "".join(f'<span class="tech-chip">{t}</span>' for t in techs)
    st.markdown(chips, unsafe_allow_html=True)

    st.markdown("### 📊 Pipeline")
    st.markdown(
        """
        ```
        Audio (wav/mp3/mic)
              ↓
        Librosa  ──→  16 kHz mono
              ↓
        Chunking  (≤ 28 s)
              ↓
        WhisperProcessor
              ↓
        WhisperForConditionalGeneration
              ↓
        Texte en Arabe Tunisien
        ```
        """
    )

    st.markdown("---")
    st.caption("Made with ❤️ in Tunisia 🇹🇳")


# ─────────────────────────────────────────────
#  HERO
# ─────────────────────────────────────────────

st.markdown(
    """
    <div class="hero">
        <h1>🎙️ Reconnaissance de la Parole Tunisienne</h1>
        <p>Comparez deux modèles Whisper fine-tunés sur le dialecte tunisien — uploadez un fichier ou enregistrez votre voix.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
#  CHARGEMENT DES MODÈLES (avec spinner)
# ─────────────────────────────────────────────

with st.spinner("⏳ Chargement des modèles… (première fois uniquement)"):
    try:
        processor_v1, model_v1, device_v1 = load_model(MODEL_V1_PATH)
        processor_v2, model_v2, device_v2 = load_model(MODEL_V2_PATH)
        models_loaded = True
    except Exception as e:
        st.error(
            f"❌ Impossible de charger les modèles : `{e}`\n\n"
            f"Vérifiez que les chemins `MODEL_V1_PATH` et `MODEL_V2_PATH` "
            f"sont corrects en haut du fichier `app.py`."
        )
        models_loaded = False

# ─────────────────────────────────────────────
#  ENTRÉE AUDIO
# ─────────────────────────────────────────────

st.markdown('<p class="section-title">📥 Source Audio</p>', unsafe_allow_html=True)

tab_upload, tab_mic = st.tabs(["📂 Upload fichier", "🎤 Enregistrement micro"])

audio_bytes: bytes | None = None

# ── Onglet Upload ──
with tab_upload:
    uploaded = st.file_uploader(
        "Glissez un fichier audio (.wav ou .mp3)",
        type=["wav", "mp3"],
        label_visibility="collapsed",
    )
    if uploaded:
        audio_bytes = uploaded.read()
        st.success(f"✅ Fichier chargé : **{uploaded.name}** ({len(audio_bytes)/1024:.1f} Ko)")

# ── Onglet Micro ──
with tab_mic:
    st.info("🎙️ Cliquez sur **Start** pour enregistrer, puis **Stop** pour terminer.")
    recorded = mic_recorder(
        start_prompt="▶️ Démarrer",
        stop_prompt="⏹️ Arrêter",
        just_once=False,
        use_container_width=True,
        key="mic_recorder",
    )
    if recorded and recorded.get("bytes"):
        audio_bytes = recorded["bytes"]
        st.success("✅ Enregistrement capturé avec succès !")

# ─────────────────────────────────────────────
#  LECTEUR AUDIO
# ─────────────────────────────────────────────

if audio_bytes:
    st.markdown('<p class="section-title">🔊 Écoute de l\'audio</p>', unsafe_allow_html=True)
    st.audio(audio_bytes, format="audio/wav")

# ─────────────────────────────────────────────
#  BOUTON DE TRANSCRIPTION COMPARATIVE
# ─────────────────────────────────────────────

st.markdown('<p class="section-title">🚀 Transcription Comparative</p>', unsafe_allow_html=True)

run_disabled = not (audio_bytes and models_loaded)

if st.button("⚡ Lancer la comparaison V1 vs V2", disabled=run_disabled, use_container_width=True):

    with st.spinner("🔄 Prétraitement audio…"):
        try:
            waveform = preprocess_audio(audio_bytes)
            duration = len(waveform) / SAMPLE_RATE
        except Exception as e:
            st.error(f"❌ Erreur de prétraitement : {e}")
            st.stop()

    st.info(
        f"📏 Durée : **{duration:.1f} s** | "
        f"Échantillonnage : **16 kHz** | "
        f"{'🔀 Chunking activé' if duration > CHUNK_SECONDS else '✅ Fichier court (pas de chunking)'}"
    )

    col1, col2 = st.columns(2, gap="medium")

    # ── V1 Baseline ──
    with col1:
        with st.spinner("🤖 V1 — Baseline en cours…"):
            try:
                text_v1, time_v1 = transcribe(waveform, processor_v1, model_v1, device_v1)
            except Exception as e:
                text_v1, time_v1 = f"[Erreur : {e}]", 0.0

        st.markdown(
            f"""
            <div class="result-card">
                <h3>🔵 Modèle V1 — Baseline</h3>
                <div class="transcription">{text_v1 if text_v1 else "Aucun texte reconnu."}</div>
                <span class="metric-badge">⏱️ {time_v1:.2f} s d'inférence</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── V2 Robust ──
    with col2:
        with st.spinner("🤖 V2 — Robust en cours…"):
            try:
                text_v2, time_v2 = transcribe(waveform, processor_v2, model_v2, device_v2)
            except Exception as e:
                text_v2, time_v2 = f"[Erreur : {e}]", 0.0

        st.markdown(
            f"""
            <div class="result-card">
                <h3>🟢 Modèle V2 — Robust</h3>
                <div class="transcription">{text_v2 if text_v2 else "Aucun texte reconnu."}</div>
                <span class="metric-badge">⏱️ {time_v2:.2f} s d'inférence</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Analyse de vitesse ──
    st.markdown('<p class="section-title">📊 Analyse de Performance</p>', unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)

    with m1:
        st.metric("⏱️ Temps V1", f"{time_v1:.2f} s")
    with m2:
        st.metric("⏱️ Temps V2", f"{time_v2:.2f} s")
    with m3:
        diff = time_v2 - time_v1
        label = "V2 plus lent" if diff > 0 else "V2 plus rapide"
        st.metric("Δ Vitesse", f"{abs(diff):.2f} s", delta=f"{label}", delta_color="inverse")

    # RTF (Real-Time Factor)
    rtf_v1 = time_v1 / duration if duration > 0 else 0
    rtf_v2 = time_v2 / duration if duration > 0 else 0

    st.markdown(
        f"""
        <div style='background:#EEF2FF; border-radius:10px; padding:1rem 1.3rem; margin-top:.5rem;'>
            <b>📐 Real-Time Factor (RTF)</b> — ratio temps_inférence / durée_audio <br>
            🔵 V1 : <b>{rtf_v1:.3f}</b> &nbsp;|&nbsp; 🟢 V2 : <b>{rtf_v2:.3f}</b>
            &nbsp; <span style='color:#6C757D; font-size:.85rem;'>(< 1 = temps réel, idéal)</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Copie des transcriptions ──
    st.markdown('<p class="section-title">📋 Exporter les résultats</p>', unsafe_allow_html=True)
    exp1, exp2 = st.columns(2)
    with exp1:
        st.download_button(
            "⬇️ Télécharger — V1",
            data=text_v1,
            file_name="transcription_v1.txt",
            mime="text/plain",
            use_container_width=True,
        )
    with exp2:
        st.download_button(
            "⬇️ Télécharger — V2",
            data=text_v2,
            file_name="transcription_v2.txt",
            mime="text/plain",
            use_container_width=True,
        )

elif not audio_bytes:
    st.info("👆 Uploadez un fichier audio ou enregistrez votre voix pour activer la transcription.")
elif not models_loaded:
    st.warning("⚠️ Les modèles ne sont pas chargés. Vérifiez les chemins et relancez l'application.")