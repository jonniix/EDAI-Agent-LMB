from typing import Any, Dict, Optional
import whisper

def transcribe_audio(audio_path: str, model_size: str = "medium") -> Dict[str, Any]:
    """
    Trascrive l'audio con Whisper e restituisce il dict con segmenti.
    Gestisce errori e ritorna dict vuoto se fallisce.
    """
    try:
        model: Any = whisper.load_model(model_size)
        result: Dict[str, Any] = model.transcribe(audio_path, verbose=False)
        return result
    except Exception as e:
        return {"segments": [], "error": str(e)}

def transcribe_block(
    audio_path: str,
    model: Optional[Any] = None,
    model_size: str = "medium"
) -> str:
    """
    Trascrive un singolo blocco audio e restituisce il testo.
    Usa modello passato o carica Whisper.
    """
    try:
        mdl: Any = model if model is not None else whisper.load_model(model_size)
        result: Dict[str, Any] = mdl.transcribe(audio_path, verbose=False)
        return str(result.get("text", "")).strip()
    except Exception as e:
        return f"Errore trascrizione: {e}"
