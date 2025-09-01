from typing import Optional, Any, Dict
from transformers import pipeline

LANG_MAP: Dict[str, str] = {
    "inglese": "en",
    "francese": "fr",
    "spagnolo": "es"
}

def get_translator_pipeline(target_lang: str) -> Any:
    tgt: str = LANG_MAP.get(target_lang, "en")
    return pipeline("translation", model=f"Helsinki-NLP/opus-mt-it-{tgt}")

def translate_text(text: str, target_lang: str, translator: Optional[Any] = None) -> str:
    if translator is None:
        translator = get_translator_pipeline(target_lang)
    try:
        out = translator(text)
        return out[0]['translation_text']
    except Exception as e:
        return f"Errore traduzione: {e}"
        return f"Errore traduzione: {e}"
