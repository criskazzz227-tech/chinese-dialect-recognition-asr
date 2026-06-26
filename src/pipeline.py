import logging
from time import perf_counter

import torch

from .config import (
    APP,
    CONFIDENCE_THRESHOLD,
    DIALECTS,
    DID_CKPT,
    DISPLAY_NAMES,
    HOP_SEC,
    SR,
    WINDOW_SEC,
)
from .exceptions import (
    AudioProcessingError,
    ModelNotAvailableError,
    UnsupportedClassifierError,
)
from .features import cmvn, logmel
from .io_audio import load_wav, pad_or_trim, simple_vad_trim
from .model_did import DID_CNN


logger = logging.getLogger(__name__)
id2label = {i: key for i, key in enumerate(DIALECTS)}
_did_cache = {}

CLASSIFIER_OPTIONS = {
    "cnn": "Log-Mel CNN 基线",
    "whisper": "Whisper 编码器 + LR",
}


def public_model_metadata(metadata):
    result = {
        key: metadata[key]
        for key in (
            "artifact_version",
            "model_type",
            "feature_model",
            "version",
            "created_at",
            "labels",
        )
        if key in metadata
    }
    metrics = metadata.get("metrics")
    if metrics:
        result["metrics"] = {
            key: metrics[key]
            for key in ("accuracy", "macro_f1")
            if key in metrics
        }
    return result


def load_did_model(device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cache_key = str(device)
    if cache_key in _did_cache:
        return _did_cache[cache_key]

    if not DID_CKPT.exists():
        raise ModelNotAvailableError(
            f"CNN checkpoint not found: {DID_CKPT}. "
            "Run `python -m src.train_did` first."
        )

    logger.info("Loading CNN classifier device=%s", device)
    checkpoint = torch.load(DID_CKPT, map_location=device)
    model = DID_CNN(in_ch=80, num_classes=len(DIALECTS))
    model.load_state_dict(checkpoint["model"])
    model.eval().to(device)
    metadata = checkpoint.get(
        "metadata",
        {
            "model_type": "logmel_1d_cnn",
            "version": "baseline-legacy",
        },
    )
    _did_cache[cache_key] = (model, device, metadata)
    return _did_cache[cache_key]


def iter_audio_windows(audio, window_sec=WINDOW_SEC, hop_sec=HOP_SEC):
    window_len = int(window_sec * SR)
    hop_len = int(hop_sec * SR)
    if len(audio) <= window_len:
        yield audio
        return

    for start in range(0, len(audio) - window_len + 1, hop_len):
        yield audio[start : start + window_len]

    if len(audio) % hop_len:
        yield audio[-window_len:]


@torch.inference_mode()
def predict_cnn(wav_path):
    model, device, metadata = load_did_model()
    audio = simple_vad_trim(load_wav(wav_path))
    probabilities = []
    for window in iter_audio_windows(audio):
        feature = cmvn(logmel(pad_or_trim(window)))
        feature = torch.tensor(feature).unsqueeze(0).to(device)
        logits = model(feature)
        probabilities.append(
            torch.softmax(logits, dim=1)[0].cpu().numpy()
        )

    probability = sum(probabilities) / len(probabilities)
    pred_id = int(probability.argmax())
    return (
        id2label[pred_id],
        float(probability[pred_id]),
        probability,
        len(probabilities),
        metadata,
    )


def predict_dialect(wav_path, classifier=None):
    classifier = classifier or APP.default_classifier
    if classifier == "cnn":
        return predict_cnn(wav_path)
    if classifier == "whisper":
        from .whisper_classifier import predict_with_whisper

        return predict_with_whisper(wav_path)
    raise UnsupportedClassifierError(
        f"Unsupported classifier '{classifier}'. "
        f"Choose one of: {', '.join(CLASSIFIER_OPTIONS)}."
    )


def run_pipeline(
    wav_path,
    classifier=None,
    whisper_model="base",
    translate_to="en",
    run_asr=True,
):
    started_at = perf_counter()
    try:
        duration = len(load_wav(wav_path)) / SR
        dialect, confidence, probability, segments, metadata = predict_dialect(
            wav_path,
            classifier,
        )
    except (ModelNotAvailableError, UnsupportedClassifierError):
        raise
    except Exception as exc:
        logger.exception("Dialect prediction failed path=%s", wav_path)
        raise AudioProcessingError(
            "The audio could not be decoded or classified."
        ) from exc

    ranked_ids = probability.argsort()[::-1]
    top_predictions = [
        {
            "dialect": id2label[int(index)],
            "dialect_name": DISPLAY_NAMES.get(
                id2label[int(index)],
                id2label[int(index)],
            ),
            "confidence": float(probability[index]),
        }
        for index in ranked_ids[:3]
    ]

    text = ""
    translated = ""
    if run_asr:
        try:
            from .asr_whisper import transcribe
            from .translate_mt import translate_zh_to_en

            text = transcribe(wav_path, model_name=whisper_model, language=None)
            translated = (
                translate_zh_to_en(text) if translate_to == "en" else text
            )
        except Exception as exc:
            logger.exception("Speech transcription failed path=%s", wav_path)
            raise AudioProcessingError("Speech transcription failed.") from exc

    classifier_key = classifier or APP.default_classifier
    elapsed = perf_counter() - started_at
    logger.info(
        "Prediction completed classifier=%s dialect=%s confidence=%.4f "
        "duration=%.2fs elapsed=%.2fs",
        classifier_key,
        dialect,
        confidence,
        duration,
        elapsed,
    )
    return {
        "dialect": dialect,
        "dialect_name": DISPLAY_NAMES.get(dialect, dialect),
        "confidence": confidence,
        "is_reliable": confidence >= CONFIDENCE_THRESHOLD,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "prob": probability.tolist(),
        "top_predictions": top_predictions,
        "segments": segments,
        "duration_seconds": duration,
        "classifier": classifier_key,
        "classifier_name": CLASSIFIER_OPTIONS[classifier_key],
        "model_metadata": public_model_metadata(metadata),
        "asr_text": text,
        "translated_text": translated,
    }
