"""
🇹🇳 ASR Tunisien — Interface de démonstration
Projet de Fin d'Études | Whisper Fine-Tuned pour le dialecte tunisien
Design : Dark Premium Glassmorphism — Couleurs Drapeau Tunisien
"""

import os
import time
import tempfile
import datetime
import gdown

import numpy as np
import streamlit as st
import librosa
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from streamlit_mic_recorder import mic_recorder
from pydub import AudioSegment


# ─────────────────────────────────────────────
#  CONFIGURATION FFMPEG
# ─────────────────────────────────────────────
local_ffmpeg = r"C:\Users\Bilel\Downloads\ffmpeg-8.1-essentials_build\ffmpeg-8.1-essentials_build\bin\ffmpeg.exe"
if os.path.exists(local_ffmpeg):
    AudioSegment.converter = local_ffmpeg
    os.environ["PATH"] += os.pathsep + os.path.dirname(local_ffmpeg)

# ─────────────────────────────────────────────
#  TÉLÉCHARGEMENT DES MODÈLES DEPUIS DRIVE
# ─────────────────────────────────────────────
ID_V1 = "12YdIqXixSf9fDSVUXu4viysFwXu2hh3O"
ID_V2 = "1CNH-YHd1iWwABofGchZAiteRXVqIwLyU"

def download_models():
    # Dossiers cibles (on simplifie pour éviter les sous-dossiers inutiles)
    path_v1 = "whisper-v1"
    path_v2 = "whisper-v2"
    
    # Vérification de la présence du fichier principal
    if not os.path.exists(os.path.join(path_v1, "model.safetensors")):
        with st.spinner("📥 Téléchargement du Modèle V1 Baseline..."):
            gdown.download_folder(id=ID_V1, output=path_v1, quiet=False, remaining_ok=True)
            
    if not os.path.exists(os.path.join(path_v2, "model.safetensors")):
        with st.spinner("📥 Téléchargement du Modèle V2 Robust..."):
            gdown.download_folder(id=ID_V2, output=path_v2, quiet=False, remaining_ok=True)

# Exécution du téléchargement
download_models()

# ─────────────────────────────────────────────
#  CONFIGURATION GLOBALE
# ─────────────────────────────────────────────
# Comme gdown télécharge le CONTENU du dossier Drive dans 'output', 
# le chemin est directement le nom du dossier.
MODEL_V1_PATH = "whisper-v1"
MODEL_V2_PATH = "whisper-v2"

# ─────────────────────────────────────────────
#  CONFIGURATION GLOBALE
# ─────────────────────────────────────────────
MODEL_V1_PATH = "whisper-v1/whisper-small-tunisian-FINAL"
MODEL_V2_PATH = "whisper-v2/whisper-small-tunisian-V2-ROBUST"
SAMPLE_RATE   = 16_000
CHUNK_SECONDS = 28
LANGUAGE      = "ar"
TASK          = "transcribe"

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ASR Tunisien 🇹🇳",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  INITIALISATION SESSION STATE (HISTORIQUE)
# ─────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "session_count" not in st.session_state:
    st.session_state.session_count = 0

# ─────────────────────────────────────────────
#  CSS PREMIUM DARK GLASSMORPHISM
# ─────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Noto+Naskh+Arabic:wght@400;500;600&display=swap');

    /* ═══════ VARIABLES ═══════ */
    :root {
        --red:      #E63946;
        --red-glow: rgba(230,57,70,.35);
        --gold:     #F4A261;
        --bg:       #080C14;
        --bg2:      #0E1420;
        --bg3:      #141B28;
        --glass:    rgba(255,255,255,.04);
        --glass2:   rgba(255,255,255,.08);
        --border:   rgba(255,255,255,.09);
        --border2:  rgba(255,255,255,.14);
        --text:     #F0F2F8;
        --text2:    #8892A4;
        --text3:    #4A5568;
        --blue:     #60A5FA;
        --teal:     #2DD4BF;
        --ok:       #22C55E;
        --r:        14px;
        --rl:       20px;
        --sh:       0 8px 32px rgba(0,0,0,.5);
    }

    /* ═══════ FOND GLOBAL ═══════ */
    .stApp {
        background-color: var(--bg);
        background-image:
            radial-gradient(ellipse 80% 50% at 20% -10%, rgba(230,57,70,.12) 0%, transparent 60%),
            radial-gradient(ellipse 60% 40% at 80% 110%, rgba(59,130,246,.08) 0%, transparent 55%);
        font-family: 'Outfit', sans-serif;
    }
    .stApp > div, [data-testid="stAppViewContainer"],
    [data-testid="stHeader"] { background: transparent !important; }
    .main .block-container { padding-top: 1.5rem; max-width: 1200px; }

    /* ═══════ HERO ═══════ */
    .hero-wrap {
        position: relative; overflow: hidden;
        background: var(--bg3);
        border: 1px solid var(--border2);
        border-radius: var(--rl);
        padding: 2.8rem 3rem;
        margin-bottom: 2rem;
        box-shadow: var(--sh), inset 0 1px 0 rgba(255,255,255,.06);
    }
    .hero-wrap::before {
        content:''; position:absolute; top:-80px; left:-80px;
        width:300px; height:300px;
        background:radial-gradient(circle,rgba(230,57,70,.22) 0%,transparent 70%);
        pointer-events:none;
    }
    .hero-wrap::after {
        content:''; position:absolute; bottom:-60px; right:-40px;
        width:250px; height:250px;
        background:radial-gradient(circle,rgba(59,130,246,.14) 0%,transparent 70%);
        pointer-events:none;
    }
    .hero-eyebrow {
        display:inline-flex; align-items:center; gap:6px;
        background:rgba(230,57,70,.15); border:1px solid rgba(230,57,70,.3);
        border-radius:999px; padding:3px 14px;
        font-size:.75rem; font-weight:700; color:var(--red);
        letter-spacing:.08em; text-transform:uppercase; margin-bottom:1rem;
    }
    .hero-title {
        font-size:2.5rem; font-weight:800; color:var(--text);
        margin:0 0 .7rem; line-height:1.15; letter-spacing:-.02em;
    }
    .hero-title span {
        background:linear-gradient(90deg,var(--red) 0%,var(--gold) 100%);
        -webkit-background-clip:text; -webkit-text-fill-color:transparent;
        background-clip:text;
    }
    .hero-sub { font-size:1rem; color:var(--text2); margin:0; line-height:1.6; max-width:560px; }
    .hero-stats { display:flex; gap:2.5rem; margin-top:2rem; flex-wrap:wrap; }
    .hs-num { font-size:1.55rem; font-weight:800; color:var(--text); display:block; line-height:1; }
    .hs-lbl { font-size:.72rem; color:var(--text3); font-weight:600; letter-spacing:.06em; text-transform:uppercase; margin-top:4px; display:block; }

    /* ═══════ SECTION TITLE ═══════ */
    .stitle {
        display:flex; align-items:center; gap:10px;
        font-size:.75rem; font-weight:700; color:var(--text2);
        letter-spacing:.12em; text-transform:uppercase;
        margin:2rem 0 1rem;
    }
    .stitle::before { content:''; display:inline-block; width:20px; height:2px; background:var(--red); border-radius:2px; flex-shrink:0; }
    .stitle::after  { content:''; display:block; flex:1; height:1px; background:var(--border); }

    /* ═══════ TABS ═══════ */
    [data-testid="stTabs"] [role="tablist"] {
        background:var(--bg3); border:1px solid var(--border);
        border-radius:var(--r); padding:4px; gap:4px;
    }
    [data-testid="stTabs"] [role="tab"] {
        background:transparent !important; border:none !important;
        color:var(--text2) !important; border-radius:10px !important;
        font-weight:600 !important; font-family:'Outfit',sans-serif !important;
        font-size:.88rem !important; transition:all .2s !important;
    }
    [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
        background:var(--red) !important; color:#fff !important;
        box-shadow:0 2px 10px var(--red-glow) !important;
    }
    [data-testid="stTabsContent"] {
        background:var(--bg3); border:1px solid var(--border);
        border-radius:var(--r); padding:1.5rem !important; margin-top:-1px;
    }

    /* ═══════ FILE UPLOADER ═══════ */
    [data-testid="stFileUploader"] {
        background:var(--glass) !important;
        border:2px dashed var(--border2) !important;
        border-radius:var(--r) !important; padding:1.5rem !important;
        transition:border-color .2s !important;
    }
    [data-testid="stFileUploader"]:hover { border-color:rgba(230,57,70,.5) !important; }
    [data-testid="stFileUploader"] label,
    [data-testid="stFileUploaderDropzoneInstructions"] { color:var(--text2) !important; font-family:'Outfit',sans-serif !important; }

    /* ═══════ RESULT CARDS ═══════ */
    .r-outer {
        border-radius:var(--rl); padding:1px;
        background:linear-gradient(135deg,rgba(255,255,255,.12),rgba(255,255,255,.03));
        box-shadow:var(--sh);
    }
    .r-card { background:var(--bg3); border-radius:calc(var(--rl) - 1px); padding:1.6rem 1.8rem; }
    .r-badge {
        display:inline-flex; align-items:center; gap:8px;
        border-radius:999px; padding:4px 14px 4px 8px;
        font-size:.78rem; font-weight:700; letter-spacing:.04em; text-transform:uppercase;
        margin-bottom:1rem;
    }
    .b-v1 { background:rgba(96,165,250,.12); border:1px solid rgba(96,165,250,.25); color:var(--blue); }
    .b-v2 { background:rgba(45,212,191,.12); border:1px solid rgba(45,212,191,.25); color:var(--teal); }
    .b-dot { width:7px; height:7px; border-radius:50%; display:inline-block; }
    .b-v1 .b-dot { background:var(--blue); box-shadow:0 0 7px var(--blue); }
    .b-v2 .b-dot { background:var(--teal); box-shadow:0 0 7px var(--teal); }
    .trans-box {
        background:rgba(0,0,0,.3); border:1px solid var(--border); border-radius:10px;
        padding:1.1rem 1.3rem;
        font-family:'Noto Naskh Arabic',serif; font-size:1.2rem; line-height:1.85;
        color:var(--text); direction:rtl; text-align:right; min-height:80px; word-break:break-word;
    }
    .r-meta { display:flex; align-items:center; gap:8px; margin-top:1rem; flex-wrap:wrap; }
    .chip {
        display:inline-flex; align-items:center; gap:5px;
        background:var(--glass2); border:1px solid var(--border);
        border-radius:999px; padding:4px 12px; font-size:.78rem; font-weight:600; color:var(--text2);
    }
    .chip.ok  { color:var(--ok);   border-color:rgba(34,197,94,.3); background:rgba(34,197,94,.08); }
    .chip.warn{ color:var(--gold); border-color:rgba(244,162,97,.3); background:rgba(244,162,97,.08); }

    /* ═══════ METRICS PANEL ═══════ */
    .mtr-panel {
        background:var(--bg3); border:1px solid var(--border);
        border-radius:var(--rl); padding:1.6rem 2rem; margin-top:1.5rem;
    }
    .mtr-label { font-size:.72rem; font-weight:700; color:var(--text3); letter-spacing:.1em; text-transform:uppercase; margin-bottom:1.2rem; }
    .bar-row   { margin-bottom:1rem; }
    .bar-hdr   { display:flex; justify-content:space-between; font-size:.82rem; color:var(--text2); font-weight:600; margin-bottom:6px; }
    .bar-track { height:8px; background:var(--glass2); border-radius:999px; overflow:hidden; }
    .bar-fill  { height:100%; border-radius:999px; }
    .bar-note  { font-size:.77rem; color:var(--text3); margin-top:.5rem; font-style:italic; }

    /* ═══════ BUTTONS ═══════ */
    div[data-testid="stButton"] > button {
        width:100%;
        background:linear-gradient(135deg,var(--red) 0%,#C1121F 100%) !important;
        color:#fff !important; border:none !important; border-radius:var(--r) !important;
        padding:.75rem 2rem !important; font-family:'Outfit',sans-serif !important;
        font-weight:700 !important; font-size:1rem !important;
        box-shadow:0 4px 20px var(--red-glow) !important;
        transition:all .2s !important;
    }
    div[data-testid="stButton"] > button:hover {
        transform:translateY(-2px) !important;
        box-shadow:0 8px 28px var(--red-glow) !important;
        filter:brightness(1.08) !important;
    }
    div[data-testid="stDownloadButton"] > button {
        background:var(--glass2) !important; color:var(--text) !important;
        border:1px solid var(--border2) !important; border-radius:var(--r) !important;
        font-family:'Outfit',sans-serif !important; font-weight:600 !important;
        box-shadow:none !important; transition:all .2s !important;
    }
    div[data-testid="stDownloadButton"] > button:hover {
        border-color:var(--red) !important; transform:translateY(-1px) !important;
        box-shadow:0 2px 12px var(--red-glow) !important;
    }

    /* ═══════ ALERTS ═══════ */
    [data-testid="stAlert"] {
        background:var(--glass) !important; border:1px solid var(--border) !important;
        border-radius:var(--r) !important; color:var(--text) !important;
        font-family:'Outfit',sans-serif !important;
    }

    /* ═══════ METRICS ═══════ */
    [data-testid="stMetric"] {
        background:var(--bg3) !important; border:1px solid var(--border) !important;
        border-radius:var(--r) !important; padding:1rem 1.2rem !important;
    }
    [data-testid="stMetricLabel"] {
        color:var(--text3) !important; font-family:'Outfit',sans-serif !important;
        font-size:.72rem !important; font-weight:700 !important;
        letter-spacing:.08em !important; text-transform:uppercase !important;
    }
    [data-testid="stMetricValue"] { color:var(--text) !important; font-family:'Outfit',sans-serif !important; font-weight:800 !important; }
    [data-testid="stMetricDelta"]  { font-family:'Outfit',sans-serif !important; font-weight:600 !important; font-size:.8rem !important; }

    /* ═══════ AUDIO ═══════ */
    [data-testid="stAudio"] {
        background:var(--bg3) !important; border:1px solid var(--border) !important;
        border-radius:var(--r) !important; padding:.5rem !important;
    }
    audio { border-radius:8px; filter:invert(1) hue-rotate(180deg); }

    /* ═══════ SIDEBAR ═══════ */
    [data-testid="stSidebar"] { background:var(--bg2) !important; border-right:1px solid var(--border) !important; }
    [data-testid="stSidebarContent"] { padding:1.5rem 1.2rem; }
    [data-testid="stSidebar"] .stMarkdown { color:var(--text2); }
    .sb-logo { display:flex; flex-direction:column; align-items:center; padding:1.5rem 0 1.8rem; border-bottom:1px solid var(--border); margin-bottom:1.5rem; }
    .sb-flag  { font-size:2.8rem; margin-bottom:.5rem; }
    .sb-brand { font-size:1.1rem; font-weight:800; color:var(--text); }
    .sb-sub   { font-size:.73rem; color:var(--text3); font-weight:500; margin-top:2px; }
    .sb-sec   { font-size:.7rem; font-weight:700; color:var(--text3); letter-spacing:.12em; text-transform:uppercase; margin:1.5rem 0 .7rem; }
    .tech-grid{ display:flex; flex-wrap:wrap; gap:6px; margin-bottom:1rem; }
    .tech-pill{ background:var(--glass2); border:1px solid var(--border); border-radius:999px; padding:3px 11px; font-size:.72rem; font-weight:600; color:var(--text2); }

    /* ═══════ HISTORY ═══════ */
    .h-entry {
        background:var(--bg3); border:1px solid var(--border);
        border-radius:var(--r); padding:1.2rem 1.4rem;
        margin-bottom:.8rem; position:relative; transition:border-color .2s;
    }
    .h-entry:hover { border-color:var(--border2); }
    .h-entry::before {
        content:''; position:absolute; left:0; top:50%; transform:translateY(-50%);
        width:3px; height:55%; border-radius:0 3px 3px 0;
        background:var(--red); opacity:.7;
    }
    .h-hdr { display:flex; align-items:flex-start; justify-content:space-between; margin-bottom:.75rem; gap:10px; flex-wrap:wrap; }
    .h-num  { font-size:.72rem; font-weight:700; color:var(--red); letter-spacing:.08em; text-transform:uppercase; }
    .h-ts   { font-size:.72rem; color:var(--text3); }
    .h-meta { display:flex; gap:7px; flex-wrap:wrap; margin-bottom:.8rem; }
    .h-c    { background:var(--glass); border:1px solid var(--border); border-radius:999px; padding:2px 10px; font-size:.72rem; font-weight:600; color:var(--text3); }
    .h-cols { display:grid; grid-template-columns:1fr 1fr; gap:10px; }
    .h-mlbl { font-size:.7rem; font-weight:700; text-transform:uppercase; letter-spacing:.08em; margin-bottom:5px; }
    .hv1    { color:var(--blue); }
    .hv2    { color:var(--teal); }
    .h-txt  {
        font-family:'Noto Naskh Arabic',serif; font-size:.88rem;
        direction:rtl; text-align:right; color:var(--text); line-height:1.6;
        background:rgba(0,0,0,.2); border-radius:7px; padding:.5rem .75rem;
        min-height:36px; word-break:break-word;
    }
    .h-empty { text-align:center; padding:3rem 2rem; color:var(--text3); }
    .h-empty-icon { font-size:2.5rem; opacity:.35; display:block; margin-bottom:.8rem; }

    /* ═══════ MISC ═══════ */
    #MainMenu, footer, [data-testid="manage-app-button"] { visibility:hidden; }
    hr { border-color:var(--border) !important; }
    .stMarkdown h3 { color:var(--text); font-family:'Outfit',sans-serif; font-weight:700; }
    .stMarkdown p, .stMarkdown li { color:var(--text2); font-family:'Outfit',sans-serif; }
    .stMarkdown table { border:1px solid var(--border) !important; border-radius:var(--r) !important; overflow:hidden; }
    .stMarkdown th { background:var(--glass2) !important; color:var(--text) !important; font-family:'Outfit',sans-serif !important; }
    .stMarkdown td { color:var(--text2) !important; border-color:var(--border) !important; font-family:'Outfit',sans-serif !important; }
    code { background:var(--glass2) !important; border:1px solid var(--border) !important; color:var(--gold) !important; border-radius:5px !important; }
    ::-webkit-scrollbar { width:5px; height:5px; }
    ::-webkit-scrollbar-track { background:transparent; }
    ::-webkit-scrollbar-thumb { background:var(--border2); border-radius:3px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
#  CHARGEMENT DES MODÈLES (mis en cache)
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_all_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Chargement V1 en forçant la lecture locale
    p1 = WhisperProcessor.from_pretrained(MODEL_V1_PATH, local_files_only=True)
    m1 = WhisperForConditionalGeneration.from_pretrained(MODEL_V1_PATH, local_files_only=True).to(device)
    m1.eval()
    
    # Chargement V2 en forçant la lecture locale
    p2 = WhisperProcessor.from_pretrained(MODEL_V2_PATH, local_files_only=True)
    m2 = WhisperForConditionalGeneration.from_pretrained(MODEL_V2_PATH, local_files_only=True).to(device)
    m2.eval()
    
    return p1, m1, p2, m2, device

# --- Initialisation et appel du chargement ---
models_loaded = False
try:
    with st.spinner("⏳ Chargement des modèles en mémoire (Lecture locale)..."):
        processor_v1, model_v1, processor_v2, model_v2, device = load_all_models()
        models_loaded = True
except Exception as e:
    st.error(f"❌ Erreur lors du chargement des modèles : {e}")
    st.info("💡 Vérifiez que le téléchargement depuis Drive s'est bien terminé juste au-dessus.")
# ─────────────────────────────────────────────
#  FONCTIONS AUDIO
# ─────────────────────────────────────────────

def preprocess_audio(audio_bytes: bytes):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        waveform, _ = librosa.load(tmp_path, sr=SAMPLE_RATE, mono=True)
        if len(waveform) > 0:
            peak = np.max(np.abs(waveform))
            if peak > 0:
                waveform = waveform / peak
        return waveform.astype(np.float32)
    except Exception as e:
        st.error(f"❌ Erreur de prétraitement audio : {e}")
        return None
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def chunk_audio(waveform: np.ndarray, chunk_sec: int = CHUNK_SECONDS) -> list:
    chunk_len = chunk_sec * SAMPLE_RATE
    return [waveform[i: i + chunk_len] for i in range(0, len(waveform), chunk_len)]


def transcribe(waveform: np.ndarray, processor, model, dev: str):
    chunks = chunk_audio(waveform)
    texts = []
    forced_ids = processor.get_decoder_prompt_ids(language=LANGUAGE, task=TASK)
    t0 = time.perf_counter()
    for chunk in chunks:
        inputs = processor(chunk, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features.to(dev)
        with torch.no_grad():
            ids = model.generate(inputs, forced_decoder_ids=forced_ids)
        texts.append(processor.batch_decode(ids, skip_special_tokens=True)[0].strip())
    return " ".join(texts), time.perf_counter() - t0

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        """
        <div class="sb-logo">
            <span class="sb-flag">🇹🇳</span>
            <span class="sb-brand">ASR Tunisien</span>
            <span class="sb-sub">PFE — FST Tunis 2025</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sb-sec">📖 Projet</div>', unsafe_allow_html=True)
    st.markdown(
        """
        Reconnaissance automatique de la parole pour le **dialecte tunisien** (Darija).

        | Modèle | Entraînement |
        |--------|-------------|
        | **V1 Baseline** | Fine-tuning standard |
        | **V2 Robust**   | Speed + Bruit |

        Métrique : **WER** (Word Error Rate)
        """
    )

    st.markdown('<div class="sb-sec">🛠 Stack</div>', unsafe_allow_html=True)
    techs = ["Whisper-Small", "HuggingFace 🤗", "PyTorch",
             "Librosa", "TEDxTN Corpus", "Streamlit", "Python 3.11"]
    chips = '<div class="tech-grid">' + "".join(
        f'<span class="tech-pill">{t}</span>' for t in techs
    ) + "</div>"
    st.markdown(chips, unsafe_allow_html=True)

    st.markdown('<div class="sb-sec">📊 Pipeline</div>', unsafe_allow_html=True)
    st.markdown(
        """
        ```
        Audio (wav/mp3/mic)
              ↓
        Librosa → 16 kHz mono
              ↓
        Chunking (≤ 28 s)
              ↓
        WhisperProcessor
              ↓
        Génération → Texte
        ```
        """
    )

    if st.session_state.history:
        st.markdown('<div class="sb-sec">📈 Session</div>', unsafe_allow_html=True)
        total    = len(st.session_state.history)
        avg_dur  = sum(e["duration"] for e in st.session_state.history) / total
        avg_rtf1 = sum(e["rtf_v1"]   for e in st.session_state.history) / total
        ca, cb   = st.columns(2)
        ca.metric("Analyses", total)
        cb.metric("Durée moy.", f"{avg_dur:.1f}s")
        st.caption(f"RTF moyen V1 : {avg_rtf1:.3f}")

    st.markdown("---")
    st.caption("Made with ❤️ in Tunisia 🇹🇳")

# ─────────────────────────────────────────────
#  HERO BANNER
# ─────────────────────────────────────────────

n_analyses   = len(st.session_state.history)
device_lbl   = "GPU CUDA" if (models_loaded and device == "cuda") else "CPU"
status_lbl   = "✓ Prêts" if models_loaded else "✗ Erreur"

st.markdown(
    f"""
    <div class="hero-wrap">
        <div class="hero-eyebrow">● PFE — FST Tunis · 2025</div>
        <h1 class="hero-title">🎙️ Reconnaissance <span>Vocale Tunisienne</span></h1>
        <p class="hero-sub">
            Comparez deux modèles Whisper fine-tunés sur le dialecte tunisien (Darija) —
            uploadez un fichier ou enregistrez votre voix en direct.
        </p>
        <div class="hero-stats">
            <div><span class="hs-num">{n_analyses}</span><span class="hs-lbl">Analyses session</span></div>
            <div><span class="hs-num">2</span><span class="hs-lbl">Modèles actifs</span></div>
            <div><span class="hs-num">{device_lbl}</span><span class="hs-lbl">Hardware</span></div>
            <div><span class="hs-num">{status_lbl}</span><span class="hs-lbl">État modèles</span></div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
#  ENTRÉE AUDIO
# ─────────────────────────────────────────────

st.markdown('<div class="stitle">Source Audio</div>', unsafe_allow_html=True)

tab_upload, tab_mic = st.tabs(["📂  Upload fichier", "🎤  Enregistrement micro"])

audio_bytes = None
audio_label = "audio"

with tab_upload:
    uploaded = st.file_uploader(
        "Glissez un fichier ici",
        type=["wav", "mp3"],
        label_visibility="collapsed",
    )
    if uploaded:
        audio_bytes = uploaded.read()
        audio_label = uploaded.name
        st.success(f"✅  **{uploaded.name}** — {len(audio_bytes)/1024:.1f} Ko chargés")

with tab_mic:
    st.info("🎙️  Appuyez sur **Démarrer** pour enregistrer, puis **Arrêter**.")
    recorded = mic_recorder(
        start_prompt="▶️ Démarrer",
        stop_prompt="⏹️ Arrêter",
        just_once=False,
        use_container_width=True,
        key="mic_recorder",
    )
    if recorded and recorded.get("bytes"):
        audio_bytes = recorded["bytes"]
        audio_label = f"enregistrement_{datetime.datetime.now().strftime('%H%M%S')}.wav"
        st.success("✅  Enregistrement capturé !")

# Lecteur audio
if audio_bytes:
    st.markdown('<div class="stitle">Écoute</div>', unsafe_allow_html=True)
    st.audio(audio_bytes, format="audio/wav")

# ─────────────────────────────────────────────
#  BOUTON ANALYSE COMPARATIVE
# ─────────────────────────────────────────────

st.markdown('<div class="stitle">Analyse Comparative</div>', unsafe_allow_html=True)

run_disabled = not (audio_bytes and models_loaded)

if st.button("⚡  Lancer la comparaison V1 vs V2", disabled=run_disabled, use_container_width=True):

    # — Prétraitement —
    with st.spinner("🔄  Prétraitement de l'audio…"):
        waveform = preprocess_audio(audio_bytes)

    if waveform is None or len(waveform) == 0:
        st.error("❌  Impossible de lire l'audio. Vérifiez le fichier.")
        st.stop()

    duration = len(waveform) / SAMPLE_RATE
    chunked  = duration > CHUNK_SECONDS

    st.markdown(
        f"""
        <div style="display:flex;gap:9px;flex-wrap:wrap;margin-bottom:1.2rem;">
            <span class="chip">⏱ {duration:.1f} s</span>
            <span class="chip">📡 16 kHz mono</span>
            <span class="chip">{"🔀 Chunking activé" if chunked else "✅ Fichier court"}</span>
            <span class="chip">🖥 {device_lbl}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # — Inférence —
    text_v1, time_v1 = "[Erreur]", 0.0
    text_v2, time_v2 = "[Erreur]", 0.0

    with st.spinner("🤖  V1 Baseline — transcription…"):
        try:
            text_v1, time_v1 = transcribe(waveform, processor_v1, model_v1, device)
        except Exception as e:
            text_v1 = f"[Erreur V1 : {e}]"

    with st.spinner("🤖  V2 Robust — transcription…"):
        try:
            text_v2, time_v2 = transcribe(waveform, processor_v2, model_v2, device)
        except Exception as e:
            text_v2 = f"[Erreur V2 : {e}]"

    rtf_v1  = time_v1 / duration if duration > 0 else 0
    rtf_v2  = time_v2 / duration if duration > 0 else 0
    faster  = "V1" if time_v1 < time_v2 else "V2"
    c_v1    = "ok"   if time_v1 <= time_v2 else "warn"
    c_v2    = "ok"   if time_v2 <= time_v1 else "warn"

    # — Résultats côte à côte —
    col1, col2 = st.columns(2, gap="medium")

    with col1:
        st.markdown(
            f"""
            <div class="r-outer">
              <div class="r-card">
                <div class="r-badge b-v1"><span class="b-dot"></span> V1 — Baseline</div>
                <div class="trans-box">{text_v1 or "— aucun texte reconnu —"}</div>
                <div class="r-meta">
                  <span class="chip {c_v1}">⏱ {time_v1:.2f} s</span>
                  <span class="chip">RTF {rtf_v1:.3f}</span>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="r-outer">
              <div class="r-card">
                <div class="r-badge b-v2"><span class="b-dot"></span> V2 — Robust</div>
                <div class="trans-box">{text_v2 or "— aucun texte reconnu —"}</div>
                <div class="r-meta">
                  <span class="chip {c_v2}">⏱ {time_v2:.2f} s</span>
                  <span class="chip">RTF {rtf_v2:.3f}</span>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # — Métriques st —
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("⏱ Temps V1",  f"{time_v1:.2f} s")
    mc2.metric("⏱ Temps V2",  f"{time_v2:.2f} s")
    diff = abs(time_v2 - time_v1)
    mc3.metric("🏆 Plus rapide", faster, delta=f"−{diff:.2f} s", delta_color="normal")

    # — Barres RTF —
    mx      = max(rtf_v1, rtf_v2, 0.001)
    pv1     = min(rtf_v1 / mx * 100, 100)
    pv2     = min(rtf_v2 / mx * 100, 100)

    st.markdown(
        f"""
        <div class="mtr-panel">
          <div class="mtr-label">📐 Real-Time Factor — durée inférence / durée audio</div>
          <div class="bar-row">
            <div class="bar-hdr"><span>🔵 V1 Baseline</span><span>{rtf_v1:.3f}</span></div>
            <div class="bar-track"><div class="bar-fill" style="width:{pv1:.1f}%;background:#60A5FA;"></div></div>
          </div>
          <div class="bar-row">
            <div class="bar-hdr"><span>🟢 V2 Robust</span><span>{rtf_v2:.3f}</span></div>
            <div class="bar-track"><div class="bar-fill" style="width:{pv2:.1f}%;background:#2DD4BF;"></div></div>
          </div>
          <div class="bar-note">RTF &lt; 1 → traitement plus rapide que le temps réel (idéal).</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # — Export —
    st.markdown('<div class="stitle">Export</div>', unsafe_allow_html=True)
    e1, e2 = st.columns(2)
    with e1:
        st.download_button("⬇️  Transcription V1 (.txt)", data=text_v1,
                           file_name="transcription_v1.txt", mime="text/plain",
                           use_container_width=True)
    with e2:
        st.download_button("⬇️  Transcription V2 (.txt)", data=text_v2,
                           file_name="transcription_v2.txt", mime="text/plain",
                           use_container_width=True)

    # ── SAUVEGARDE HISTORIQUE ──────────────────────────────────────────────
    st.session_state.session_count += 1
    st.session_state.history.insert(0, {
        "id":        st.session_state.session_count,
        "timestamp": datetime.datetime.now().strftime("%d/%m/%Y  %H:%M:%S"),
        "label":     audio_label,
        "duration":  duration,
        "chunked":   chunked,
        "text_v1":   text_v1,
        "time_v1":   time_v1,
        "rtf_v1":    rtf_v1,
        "text_v2":   text_v2,
        "time_v2":   time_v2,
        "rtf_v2":    rtf_v2,
        "faster":    faster,
    })
    st.rerun()

elif not audio_bytes:
    st.markdown(
        """
        <div style="text-align:center;padding:3rem 2rem;
            border:2px dashed rgba(255,255,255,.07);border-radius:16px;
            color:rgba(255,255,255,.2);font-size:.93rem;font-weight:500;">
            ⬆️  Uploadez un fichier audio ou enregistrez votre voix pour activer l'analyse
        </div>
        """,
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────
#  HISTORIQUE DE SESSION
# ─────────────────────────────────────────────

st.markdown('<div class="stitle">Historique de la session</div>', unsafe_allow_html=True)

if not st.session_state.history:
    st.markdown(
        """
        <div class="h-empty">
            <span class="h-empty-icon">🕐</span>
            <div style="font-size:.88rem;font-weight:500;color:var(--text3);">
                Aucune analyse pour l'instant.<br>Lancez votre première transcription ci-dessus.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    clr_col, _ = st.columns([1, 4])
    with clr_col:
        if st.button("🗑️  Effacer l'historique", use_container_width=True):
            st.session_state.history = []
            st.session_state.session_count = 0
            st.rerun()

    for e in st.session_state.history:
        v1_chip = "ok"   if e["time_v1"] <= e["time_v2"] else "warn"
        v2_chip = "ok"   if e["time_v2"] <= e["time_v1"] else "warn"
        st.markdown(
            f"""
            <div class="h-entry">
                <div class="h-hdr">
                    <span class="h-num">Analyse #{e['id']}</span>
                    <span class="h-ts">🕐 {e['timestamp']}</span>
                </div>
                <div class="h-meta">
                    <span class="h-c">📁 {e['label']}</span>
                    <span class="h-c">⏱ {e['duration']:.1f} s</span>
                    <span class="h-c">{"🔀 Chunké" if e['chunked'] else "✅ Court"}</span>
                    <span class="h-c">🏆 {e['faster']} plus rapide</span>
                    <span class="h-c">V1 : {e['time_v1']:.2f}s · RTF {e['rtf_v1']:.3f}</span>
                    <span class="h-c">V2 : {e['time_v2']:.2f}s · RTF {e['rtf_v2']:.3f}</span>
                </div>
                <div class="h-cols">
                    <div>
                        <div class="h-mlbl hv1">V1 Baseline</div>
                        <div class="h-txt">{e['text_v1'] or '—'}</div>
                    </div>
                    <div>
                        <div class="h-mlbl hv2">V2 Robust</div>
                        <div class="h-txt">{e['text_v2'] or '—'}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )