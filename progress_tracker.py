from typing import Optional

class ProgressTracker:
    """
    Traccia avanzamento della lavorazione audio a blocchi.
    Calcola MB elaborati e tempo stimato rimanente.
    """
    def __init__(self, total_chunks: int, total_bytes: int) -> None:
        self.total_chunks: int = total_chunks
        self.total_bytes: int = total_bytes
        self.chunk_bytes: float = total_bytes / total_chunks if total_chunks > 0 else 1.0
        self.current_chunk: int = 0

    def update(self, current_chunk: int) -> None:
        self.current_chunk = current_chunk

    def mb_processed(self, current_chunk: Optional[int] = None) -> float:
        idx: int = current_chunk if current_chunk is not None else self.current_chunk
        return (idx * self.chunk_bytes) / (1024 * 1024)

    def estimate_remaining(self, elapsed: float) -> str:
        if self.current_chunk == 0:
            return "Stima non disponibile"
        try:
            avg: float = elapsed / self.current_chunk
            remain: float = avg * (self.total_chunks - self.current_chunk)
            mins, secs = divmod(int(remain), 60)
            return f"{mins}m {secs}s"
        except Exception:
            return "Stima non disponibile"
