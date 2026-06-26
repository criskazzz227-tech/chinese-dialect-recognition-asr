import numpy as np
import librosa

from .config import HOP_LENGTH, N_FFT, N_MELS, SR, WIN_LENGTH


def logmel(x: np.ndarray):
    """Return log-Mel features with shape [n_mels, T]."""
    spectrum = librosa.feature.melspectrogram(
        y=x,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        n_mels=N_MELS,
        power=2.0,
    )
    spectrum = np.log(np.maximum(spectrum, 1e-10))
    return spectrum.astype(np.float32)


def mfcc(x: np.ndarray, n_mfcc: int = 20):
    """Return MFCC plus delta and delta-delta features."""
    base = librosa.feature.mfcc(
        y=x,
        sr=SR,
        n_mfcc=n_mfcc,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        n_mels=N_MELS,
    )
    d1 = librosa.feature.delta(base)
    d2 = librosa.feature.delta(base, order=2)
    feat = np.concatenate([base, d1, d2], axis=0)
    return feat.astype(np.float32)


def cmvn(feat: np.ndarray):
    """Apply cepstral mean and variance normalization."""
    mean = feat.mean(axis=1, keepdims=True)
    std = feat.std(axis=1, keepdims=True) + 1e-6
    return (feat - mean) / std
