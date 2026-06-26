import numpy as np
import pytest

from src.exceptions import UnsupportedClassifierError
from src.pipeline import iter_audio_windows, predict_dialect


def test_audio_windowing_covers_short_and_long_audio():
    short = np.zeros(16000, dtype=np.float32)
    long = np.zeros(16000 * 10, dtype=np.float32)

    assert len(list(iter_audio_windows(short))) == 1
    assert len(list(iter_audio_windows(long))) >= 2


def test_unknown_classifier_is_rejected():
    with pytest.raises(UnsupportedClassifierError):
        predict_dialect("unused.wav", classifier="unknown")
