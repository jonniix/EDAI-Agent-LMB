from typing import List, Dict, Any
import whisperx

def diarize_speakers(audio_path: str, transcription: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Esegue diarizzazione con whisperx e associa i segmenti trascritti ai parlanti.
    Gestisce errori e ritorna una lista di dict con speaker e testo.
    """
    diarized_segments: List[Dict[str, str]] = []
    try:
        model = whisperx.load_model("medium", device="cpu")
        diarize_model = whisperx.DiarizationPipeline(use_auth_token=None, device="cpu")
        diarize_result: Dict[str, Any] = diarize_model(audio_path)
        segments: List[Dict[str, Any]] = transcription.get("segments", [])
        diarize_segments: List[Dict[str, Any]] = diarize_result.get("segments", [])
        for seg in segments:
            speaker: str = "Voce ?"
            seg_start: float = float(seg.get("start", 0.0))
            seg_end: float = float(seg.get("end", 0.0))
            for dseg in diarize_segments:
                dseg_start: float = float(dseg.get("start", 0.0))
                dseg_end: float = float(dseg.get("end", 0.0))
                if seg_start >= dseg_start and seg_end <= dseg_end:
                    speaker = f"Voce {int(dseg.get('speaker', 0)) + 1}"
                    break
            diarized_segments.append({
                "speaker": speaker,
                "text": str(seg.get("text", "")).strip()
            })
    except Exception as e:
        # In caso di errore, restituisci tutto come Voce 1
        diarized_segments = [
            {"speaker": "Voce 1", "text": str(seg.get("text", "")).strip()}
            for seg in transcription.get("segments", [])
        ]
    return diarized_segments
    return diarized_segments
