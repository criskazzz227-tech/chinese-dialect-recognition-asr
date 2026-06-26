import logging
from datetime import datetime, timezone

import joblib
import numpy as np
import torch
import whisper

from .config import (
    APP,
    AUDIO,
    DIALECTS,
    WHISPER_CLASSIFIER_PATH,
)
from .exceptions import ModelNotAvailableError
from .io_audio import load_wav, pad_or_trim, simple_vad_trim


logger = logging.getLogger(__name__)
_encoder_cache = {}
_classifier_cache = None


def load_whisper_encoder(model_name=None, device=None):
    model_name = model_name or APP.whisper_feature_model
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cache_key = (model_name, str(device))
    if cache_key not in _encoder_cache:
        logger.info("Loading Whisper encoder model=%s device=%s", model_name, device)
        model = whisper.load_model(model_name, device=device)
        model.eval()
        _encoder_cache[cache_key] = model
    return _encoder_cache[cache_key], device


@torch.inference_mode()
def extract_whisper_embedding(audio_path, model=None, device=None):
    audio = simple_vad_trim(load_wav(audio_path))
    return extract_whisper_embedding_from_audio(audio, model, device)


@torch.inference_mode()
def extract_whisper_embedding_from_audio(audio, model=None, device=None):
    if model is None:
        model, device = load_whisper_encoder(device=device)
    else:
        device = device or next(model.parameters()).device

    audio = pad_or_trim(audio, max_len=AUDIO.max_samples)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(
        audio,
        n_mels=model.dims.n_mels,
        device=device,
    )
    encoded = model.encoder(mel.unsqueeze(0))
    pooled = torch.cat(
        [encoded.mean(dim=1), encoded.std(dim=1)],
        dim=1,
    )
    return pooled.squeeze(0).float().cpu().numpy()


def save_classifier(pipeline, metrics, feature_model):
    artifact = {
        "artifact_version": 1,
        "model_type": "whisper_encoder_logistic_regression",
        "feature_model": feature_model,
        "labels": DIALECTS,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
        "pipeline": pipeline,
    }
    joblib.dump(artifact, WHISPER_CLASSIFIER_PATH)
    logger.info("Saved Whisper classifier to %s", WHISPER_CLASSIFIER_PATH)


def load_classifier():
    global _classifier_cache
    if _classifier_cache is not None:
        return _classifier_cache
    if not WHISPER_CLASSIFIER_PATH.exists():
        raise ModelNotAvailableError(
            "Whisper feature classifier is not trained. "
            "Run `python -m src.train_whisper_classifier` first."
        )
    _classifier_cache = joblib.load(WHISPER_CLASSIFIER_PATH)
    return _classifier_cache


def predict_with_whisper(audio_path):
    artifact = load_classifier()
    model, device = load_whisper_encoder(artifact["feature_model"])
    audio = simple_vad_trim(load_wav(audio_path))
    window_len = int(AUDIO.window_seconds * AUDIO.sample_rate)
    hop_len = int(AUDIO.hop_seconds * AUDIO.sample_rate)
    windows = []
    if len(audio) <= window_len:
        windows.append(audio)
    else:
        windows.extend(
            audio[start : start + window_len]
            for start in range(0, len(audio) - window_len + 1, hop_len)
        )
        if len(audio) % hop_len:
            windows.append(audio[-window_len:])

    embeddings = np.stack(
        [
            extract_whisper_embedding_from_audio(
                window,
                model=model,
                device=device,
            )
            for window in windows
        ]
    )
    probabilities = artifact["pipeline"].predict_proba(embeddings).mean(axis=0)

    probability_by_label = {
        label: float(score)
        for label, score in zip(
            artifact["pipeline"].classes_,
            probabilities,
        )
    }
    ordered = np.array(
        [probability_by_label.get(label, 0.0) for label in DIALECTS],
        dtype=np.float32,
    )
    pred_id = int(ordered.argmax())
    return (
        DIALECTS[pred_id],
        float(ordered[pred_id]),
        ordered,
        len(windows),
        artifact,
    )
