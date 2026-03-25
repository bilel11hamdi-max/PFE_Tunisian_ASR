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
import gdown
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from streamlit_mic_recorder import mic_recorder
from pydub import AudioSegment

# ─────────────────────────────────────────────
#  CONFIGURATION FFMPEG (Hébergement Local vs Cloud)
# ─────────────────────────────────────────────
# Sur Windows en local, on force le chemin. Sur Streamlit Cloud, FFmpeg s'installe via packages.txt
local_ffmpeg = r"C:\Users\Bilel\Downloads\ffmpeg-8.1-essentials_build\ffmpeg-8.1-essentials_build\bin\ffmpeg.exe"

if os.path.exists(local_ffmpeg):
    AudioSegment.converter = local_ffmpeg
    os.environ["PATH"] += os.pathsep + os.path.dirname(local_ffmpeg)


# ─────────────────────────────────────────────
#  TÉLÉCHARGEMENT DES MODÈLES DEPUIS DRIVE
# ─────────────────────────────────────────────
# On ne garde que l'ID propre pour gdown
ID_V1 = "12YdIqXixSf9fDSVUXu4viysFwXu2hh3O"
ID_V2 = "1CNH-YHd1iWwABofGchZAiteRXVqIwLyU"

def download_models():
    if not os.path.exists("whisper-v1"):
        st.info("Téléchargement du Modèle V1 Baseline (première fois)...")
        gdown.download_folder(id=ID_V1, output="whisper-v1", quiet=False)
    if not os.path.exists("whisper-v2"):
        st.info("Téléchargement du Modèle V2 Fine-tuné (première fois)...")
        gdown.download_folder(id=ID_V2, output="whisper-v2", quiet=False)

download_models()


# ─────────────────────────────────────────────
#  CONFIGURATION GLOBALE
# ─────────────────────────────────────────────
MODEL_V1_PATH = "whisper-v1"
MODEL_V2_PATH = "whisper-v2"

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

    .stApp { background-color: var(--bg); }

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

    .section-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: var(--secondary);
        border-left: 4px solid var(--primary);
        padding-left: .7rem;
        margin: 1.5rem 0 .8rem;
    }

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
    
    #MainMenu, footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────
#  CHARGEMENT DES MODÈLES (optimisé et mis en cache)
# ─────────────────────────────────────────────

MODEL_V1_PATH = "whisper-v1"
MODEL_V2_PATH = "whisper-v2"

@st.cache_resource(show_spinner=False)
def load_all_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    p1 = WhisperProcessor.from_pretrained(MODEL_V1_PATH)
    m1 = WhisperForConditionalGeneration.from_pretrained(MODEL_V1_PATH).to(device)
    m1.eval()
    
    p2 = WhisperProcessor.from_pretrained(MODEL_V2_PATH)
    m2 = WhisperForConditionalGeneration.from_pretrained(MODEL_V2_PATH).to(device)
    m2.eval()
    
    return p1, m1, p2, m2, device

# Appel du chargement unique
with st.spinner("⏳ Chargement initial des modèles en mémoire (S'il vous plaît patientez)..."):
    processor_v1, model_v1, processor_v2, model_v2, device = load_all_models()
    models_loaded = True


# ─────────────────────────────────────────────
#  PRÉTRAITEMENT AUDIO
# ─────────────────────────────────────────────

def preprocess_audio(audio_bytes: bytes):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    
    try:
        waveform, sr = librosa.load(tmp_path, sr=16000, mono=True)
        if len(waveform) > 0:
            peak = np.max(np.abs(waveform))
            if peak > 0:
                waveform = waveform / peak
        return waveform
    except Exception as e:
        st.error(f"Erreur lors du traitement : {e}")
        return None
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def chunk_audio(waveform: np.ndarray, chunk_sec: int = CHUNK_SECONDS) -> list[np.ndarray]:
    chunk_len = chunk_sec * SAMPLE_RATE
    return [waveform[i : i + chunk_len] for i in range(0, len(waveform), chunk_len)]


# ─────────────────────────────────────────────
#  FONCTION D'INFÉRENCE
# ─────────────────────────────────────────────

def transcribe(waveform: np.ndarray, processor, model, device: str) -> tuple[str, float]:
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
#  SIDEBAR (Menu latéral)
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
#  HERO BANNER
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
        
        audio_waveform = preprocess_audio(audio_bytes)

        if audio_waveform is not None:
            with st.spinner("🤖 Transcription directe en cours..."):
                # Prétraitement et forçage de langue pour le micro direct
                forced_ids = processor_v1.get_decoder_prompt_ids(language=LANGUAGE, task=TASK)

                # V1
                input_v1 = processor_v1(audio_waveform, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features.to(device)
                with torch.no_grad():
                    ids_v1 = model_v1.generate(input_v1, forced_decoder_ids=forced_ids)
                text_v1_mic = processor_v1.batch_decode(ids_v1, skip_special_tokens=True)[0]

                # V2
                input_v2 = processor_v2(audio_waveform, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features.to(device)
                with torch.no_grad():
                    ids_v2 = model_v2.generate(input_v2, forced_decoder_ids=forced_ids)
                text_v2_mic = processor_v2.batch_decode(ids_v2, skip_special_tokens=True)[0]
                
                # Affichage direct sous l'enregistrement
                c1, c2 = st.columns(2)
                with c1:
                    st.info("### V1 (Baseline)")
                    st.write(text_v1_mic)
                with c2:
                    st.success("### V2 (Fine-tuné)")
                    st.write(text_v2_mic)
        else:
            st.error("Erreur : Impossible d'analyser l'audio. Vérifie tes fichiers.")


# ─────────────────────────────────────────────
#  LECTEUR AUDIO GLOBAL
# ─────────────────────────────────────────────

if audio_bytes:
    st.markdown('<p class="section-title">🔊 Écoute de l\'audio</p>', unsafe_allow_html=True)
    st.audio(audio_bytes, format="audio/wav")


# ─────────────────────────────────────────────
#  BOUTON DE TRANSCRIPTION ET ANALYSE COMPARATIVE
# ─────────────────────────────────────────────

st.markdown('<p class="section-title">🚀 Analyse Comparative Globale</p>', unsafe_allow_html=True)

run_disabled = not (audio_bytes and models_loaded)

if st.button("⚡ Lancer l'Analyse Comparative V1 vs V2", disabled=run_disabled, use_container_width=True):

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
        f"{'🔀 Chunking (découpage) activé' if duration > CHUNK_SECONDS else '✅ Fichier court (pas de découpage)'}"
    )

    col1, col2 = st.columns(2, gap="medium")

    # ── V1 Baseline ──
    with col1:
        with st.spinner("🤖 V1 — Baseline en cours…"):
            try:
                text_v1, time_v1 = transcribe(waveform, processor_v1, model_v1, device)
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
                text_v2, time_v2 = transcribe(waveform, processor_v2, model_v2, device)
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
    st.markdown('<p class="section-title">📊 Métriques de Performance</p>', unsafe_allow_html=True)

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

    # ── Téléchargements ──
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
    st.info("👆 Uploadez un fichier audio ou enregistrez votre voix pour activer l'analyse.")