import streamlit as st
import uuid
import tempfile
import subprocess
import os
from typing import List
from pydub import AudioSegment
from pydub.utils import mediainfo


def save_uploaded_file(uploaded_file) -> str:
    """
    Salva un file audio caricato come file temporaneo e lo converte in WAV.
    Prova prima con ffmpeg, se fallisce usa pydub come fallback.
    """
    ext = os.path.splitext(uploaded_file.name)[-1]
    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    try:
        temp_input.write(uploaded_file.read())
        temp_input.close()

        wav_path: str = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4().hex}.wav")

        # Tentativo con ffmpeg
        ffmpeg_cmd: List[str] = ["ffmpeg", "-y", "-i", temp_input.name, wav_path]
        try:
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
            if result.returncode == 0 and os.path.exists(wav_path):
                return wav_path
            st.warning("⚠️ ffmpeg ha fallito. Provo con metodo alternativo...")
            st.code(result.stderr)
        except FileNotFoundError as e:
            st.error("❌ ffmpeg non è installato sul sistema.")
            st.stop()

        # Fallback con pydub
        try:
            audio = AudioSegment.from_file(temp_input.name)
            audio.export(wav_path, format="wav")
            if os.path.exists(wav_path):
                return wav_path
            raise RuntimeError("Export WAV fallito con pydub")
        except Exception as e:
            st.error(f"❌ Il file audio non è supportato o è danneggiato. Dettagli: {e}")
            st.stop()

    finally:
        try:
            os.remove(temp_input.name)
        except Exception:
            pass


def split_audio_in_chunks(audio_path: str, seconds: int = 20) -> List[str]:
    """
    Divide un file audio in blocchi della durata specificata.
    Ogni blocco è salvato come file temporaneo WAV.
    """
    try:
        info = mediainfo(audio_path)
        duration: float = float(info.get("duration", 0))
    except Exception as e:
        st.error(f"❌ Impossibile leggere la durata del file: {e}")
        return []

    chunk_paths: List[str] = []
    num_chunks = int(duration // seconds) + (1 if duration % seconds > 0 else 0)

    for i in range(num_chunks):
        start: int = i * seconds
        chunk_filename = f"{uuid.uuid4().hex}_chunk.wav"
        chunk_path = os.path.join(tempfile.gettempdir(), chunk_filename)

        cmd: List[str] = [
            "ffmpeg", "-y", "-i", audio_path,
            "-ss", str(start), "-t", str(seconds),
            chunk_path
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0 and os.path.exists(chunk_path):
                chunk_paths.append(chunk_path)
            else:
                st.warning(f"⚠️ Errore nel creare il blocco {i+1}:")
                st.code(result.stderr)
        except Exception as e:
            st.error(f"❌ Errore durante split ffmpeg per blocco {i+1}: {e}")

    return chunk_paths


# Test di base
def test_save_uploaded_file() -> None:
    import io
    fake = io.BytesIO(b"RIFF....WAVEfmt ")
    fake.name = "fake.wav"
    out_path = save_uploaded_file(fake)
    assert out_path.endswith(".wav")
    assert os.path.exists(out_path)


def test_split_audio_in_chunks() -> None:
    path = os.path.join("samples", "test_audio.wav")  # Aggiungi un file test reale se vuoi testare
    if os.path.exists(path):
        chunks = split_audio_in_chunks(path, seconds=5)
        assert isinstance(chunks, list)
        assert all(os.path.exists(p) for p in chunks)
