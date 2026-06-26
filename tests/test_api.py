from fastapi.testclient import TestClient

import app as api_module


client = TestClient(api_module.app)


def test_health_exposes_version_and_models():
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert "cnn" in payload["classifiers"]
    assert "whisper" in payload["classifiers"]


def test_predict_rejects_unsupported_extension():
    response = client.post(
        "/predict",
        files={"file": ("notes.txt", b"not audio", "text/plain")},
    )
    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "unsupported_audio_format"


def test_predict_rejects_empty_file():
    response = client.post(
        "/predict",
        files={"file": ("empty.wav", b"", "audio/wav")},
    )
    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "empty_file"


def test_predict_response_schema(monkeypatch):
    def fake_pipeline(*args, **kwargs):
        return {
            "dialect": "shanghai",
            "dialect_name": "上海话",
            "confidence": 0.8,
            "is_reliable": True,
            "confidence_threshold": 0.55,
            "prob": [0.8, 0.05, 0.05, 0.05, 0.05],
            "top_predictions": [
                {
                    "dialect": "shanghai",
                    "dialect_name": "上海话",
                    "confidence": 0.8,
                }
            ],
            "segments": 1,
            "duration_seconds": 3.0,
            "classifier": "whisper",
            "classifier_name": "Whisper 编码器 + LR",
            "model_metadata": {"model_type": "test"},
            "asr_text": "",
            "translated_text": "",
        }

    monkeypatch.setattr(api_module, "run_pipeline", fake_pipeline)
    response = client.post(
        "/predict?classifier=whisper&run_asr=false",
        files={"file": ("sample.wav", b"fake wav bytes", "audio/wav")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["classifier"] == "whisper"
    assert payload["probabilities"]["上海话"] == 0.8
