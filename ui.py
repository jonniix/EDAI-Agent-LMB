import streamlit as st
import os
import time
from typing import List, Optional, Dict
from transcriber import transcribe_block
from translator import get_translator_pipeline, translate_text
from summarizer import summarize_text
from utils.audio_tools import save_uploaded_file, split_audio_in_chunks

OUTPUT_TRASCR: str = os.path.join("output", "trascrizioni")
OUTPUT_RIASS: str = os.path.join("output", "riassunti")
CHUNK_SECONDS: int = 20

LANG_FLAGS: Dict[str, str] = {
    "inglese": "🇬🇧",
    "francese": "🇫🇷",
    "spagnolo": "🇪🇸"
}

@st.cache_resource
def get_whisper_model(model_size: str = "medium"):
    import whisper
    return whisper.load_model(model_size)

@st.cache_resource
def get_translator_cached(target_lang: str):
    return get_translator_pipeline(target_lang)

def run() -> None:
    st.set_page_config(page_title="Generatore Verbale Professionale", layout="wide")

    st.markdown("""
        <style>
        .big-title {font-size:2.2em;font-weight:700;margin-bottom:0.2em;}
        .subtitle {font-size:1.2em;color:#555;}
        .a4-sheet {
            background: white;
            padding: 2em;
            border: 1px solid #ddd;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            min-height: 400px;
            font-family: 'Georgia', serif;
            line-height: 1.5;
            white-space: pre-wrap;
        }
        </style>
    """, unsafe_allow_html=True)

    st.sidebar.title("🎚️ Impostazioni")
    uploaded_file = st.sidebar.file_uploader("Carica file audio", type=["mp3", "wav", "m4a", "aac"])
    include_translation = st.sidebar.checkbox("Traduci testo", value=True)
    translation_lang = st.sidebar.selectbox("Lingua traduzione", list(LANG_FLAGS.keys()), index=0)
    animation_speed = st.sidebar.slider("Velocità scrittura", min_value=0.005, max_value=0.1, step=0.005, value=0.03)
    include_summary = st.sidebar.checkbox("Aggiungi riassunto finale", value=True)
    detail_level = st.sidebar.selectbox("Livello dettaglio riassunto", ["Breve", "Medio", "Approfondito"], index=1)

    st.markdown('<div class="big-title">📝 Generatore Verbale Live</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Trascrizione animata con traduzione blocco per blocco su layout tipo A4</div>', unsafe_allow_html=True)
    st.write("---")

    if uploaded_file is not None:
        st.toast("📥 File caricato, elaborazione in corso...")

        try:
            audio_path = save_uploaded_file(uploaded_file)
        except Exception as e:
            st.error(f"Errore salvataggio/conversione: {e}")
            return

        try:
            chunks = split_audio_in_chunks(audio_path, seconds=CHUNK_SECONDS)
        except Exception as e:
            st.error(f"Errore nella suddivisione audio: {e}")
            return

        if not chunks:
            st.warning("⚠️ Il file audio è troppo breve o danneggiato.")
            return

        whisper_model = get_whisper_model()
        translator = get_translator_cached(translation_lang) if include_translation else None
        progress = st.progress(0)
        text_area = st.empty()
        translation_area = st.empty()

        document_text = ""
        translated_text = ""
        start_time = time.perf_counter()

        for i, chunk_path in enumerate(chunks):
            try:
                transcript = transcribe_block(chunk_path, model=whisper_model)
            except Exception as e:
                st.error(f"❌ Errore trascrizione: {e}")
                continue

            for word in transcript.split():
                document_text += word + " "
                styled_text = f"<div class='a4-sheet'>{document_text}</div>"
                text_area.markdown(styled_text, unsafe_allow_html=True)
                time.sleep(animation_speed)

            if include_translation:
                try:
                    translated = translate_text(transcript, translation_lang, translator=translator)
                    for word in translated.split():
                        translated_text += word + " "
                        styled_translation = f"<div class='a4-sheet'><b>{LANG_FLAGS[translation_lang]} Traduzione:</b>\n{translated_text}</div>"
                        translation_area.markdown(styled_translation, unsafe_allow_html=True)
                        time.sleep(animation_speed)
                except Exception as e:
                    st.warning(f"⚠️ Traduzione fallita: {e}")

            percent = (i + 1) / len(chunks)
            elapsed = time.perf_counter() - start_time
            minutes, seconds = divmod(int(elapsed), 60)
            progress.progress(percent, text=f"🧩 Blocco {i+1}/{len(chunks)} –⏱️ {minutes}m {seconds}s")

        progress.progress(1.0, text="✅ Tutti i blocchi completati!")

        if include_summary:
            st.subheader("📄 Riassunto finale")
            try:
                summary = summarize_text(document_text, detail_level)
                st.markdown(f"<div class='a4-sheet'>{summary}</div>", unsafe_allow_html=True)
            except Exception as e:
                st.warning(f"⚠️ Errore riassunto: {e}")
