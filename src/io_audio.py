import numpy as np
import librosa

from .config import MAX_SAMPLES, SR


def load_wav(path: str, sr: int = SR):
    x, _ = librosa.load(path, sr=sr, mono=True)
    return x.astype(np.float32)


def simple_vad_trim(x: np.ndarray, top_db: int = 30):
    """Remove leading and trailing silence with a lightweight VAD."""
    yt, _ = librosa.effects.trim(x, top_db=top_db)
    return yt


def pad_or_trim(x: np.ndarray, max_len: int = MAX_SAMPLES):
    if len(x) >= max_len:
        return x[:max_len]
    out = np.zeros((max_len,), dtype=np.float32)
    out[:len(x)] = x
    return out
