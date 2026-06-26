import whisper


# The first call downloads the selected model if it is not cached locally.
_model_cache = {}


def transcribe(audio_path: str, model_name: str = "base", language: str = None):
    if model_name not in _model_cache:
        _model_cache[model_name] = whisper.load_model(model_name)

    model = _model_cache[model_name]
    result = model.transcribe(audio_path, language=language, fp16=False)
    return result["text"]
