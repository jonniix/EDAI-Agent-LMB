<<<<<<< HEAD
from typing import List
from transformers import pipeline

def summarize_text(text: str, detail_level: str) -> str:
    """
    Riassume il testo usando transformers/BART.
    Gestisce chunk troppo lunghi, errori e livelli di dettaglio.
    """
    max_length: int
    min_length: int
    if detail_level == "Breve":
        max_length = 80
        min_length = 30
    elif detail_level == "Approfondito":
        max_length = 300
        min_length = 120
    else:
        max_length = 150
        min_length = 60

    try:
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        chunks: List[str] = [text[i:i+1000] for i in range(0, len(text), 1000)]
        summary: str = ""
        for chunk in chunks:
            try:
                out = summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)
                summary += out[0]['summary_text'] + " "
            except Exception as e:
                summary += f"[Errore riassunto chunk: {e}] "
        return summary.strip()
    except Exception as e:
        return f"Errore riassunto: {e}"
=======
from typing import List
from transformers import pipeline

def summarize_text(text: str, detail_level: str) -> str:
    """
    Riassume il testo usando transformers/BART.
    Gestisce chunk troppo lunghi, errori e livelli di dettaglio.
    """
    max_length: int
    min_length: int
    if detail_level == "Breve":
        max_length = 80
        min_length = 30
    elif detail_level == "Approfondito":
        max_length = 300
        min_length = 120
    else:
        max_length = 150
        min_length = 60

    try:
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        chunks: List[str] = [text[i:i+1000] for i in range(0, len(text), 1000)]
        summary: str = ""
        for chunk in chunks:
            try:
                out = summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)
                summary += out[0]['summary_text'] + " "
            except Exception as e:
                summary += f"[Errore riassunto chunk: {e}] "
        return summary.strip()
    except Exception as e:
        return f"Errore riassunto: {e}"
>>>>>>> 9355cb45 (chore: initial commit)
