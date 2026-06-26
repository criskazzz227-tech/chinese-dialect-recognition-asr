import logging
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated, Literal

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import APP, DIALECTS, DISPLAY_NAMES
from src.exceptions import AudioProcessingError, DialectAppError
from src.logging_config import configure_logging
from src.pipeline import CLASSIFIER_OPTIONS, run_pipeline


configure_logging()
logger = logging.getLogger(__name__)
SUPPORTED_SUFFIXES = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}


class HealthResponse(BaseModel):
    status: str
    app_version: str
    dialects: list[str]
    classifiers: dict[str, str]


class TopPrediction(BaseModel):
    dialect: str
    dialect_name: str
    confidence: float = Field(ge=0, le=1)


class PredictionResponse(BaseModel):
    dialect: str
    dialect_name: str
    confidence: float = Field(ge=0, le=1)
    is_reliable: bool
    confidence_threshold: float
    classifier: str
    classifier_name: str
    model_metadata: dict
    top_predictions: list[TopPrediction]
    probabilities: dict[str, float]
    segments: int
    duration_seconds: float
    asr_text: str
    translated_text: str


app = FastAPI(
    title=APP.name,
    summary="五类中文地域方言识别与可选语音转写 API",
    description=(
        "面向实验与作品展示的中文地域方言识别接口。"
        "当前模型仅覆盖上海、长沙、郑州、天津和南昌五类语音，"
        "不代表通用中文方言识别能力。"
    ),
    version=APP.version,
    contact={"name": "Dialect Recognition Project"},
    license_info={"name": "For research and portfolio demonstration"},
    openapi_tags=[
        {"name": "system", "description": "服务状态与模型信息"},
        {"name": "prediction", "description": "音频方言识别与转写"},
    ],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["system"],
    summary="检查服务状态",
)
def health():
    return {
        "status": "ok",
        "app_version": APP.version,
        "dialects": DIALECTS,
        "classifiers": CLASSIFIER_OPTIONS,
    }


@app.get(
    "/models",
    tags=["system"],
    summary="获取可选方言分类器",
)
def models():
    return {
        "default": APP.default_classifier,
        "classifiers": [
            {"id": key, "name": name}
            for key, name in CLASSIFIER_OPTIONS.items()
        ],
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["prediction"],
    summary="识别上传音频的地域方言",
    responses={
        400: {"description": "文件格式、文件大小或音频内容无效"},
        503: {"description": "所选模型尚未准备好"},
    },
)
async def predict(
    file: Annotated[
        UploadFile,
        File(description="WAV、MP3、M4A、FLAC 或 OGG 音频文件"),
    ],
    classifier: Annotated[
        Literal["cnn", "whisper"],
        Query(description="方言分类器：CNN 基线或 Whisper 特征分类器"),
    ] = APP.default_classifier,
    run_asr: Annotated[
        bool,
        Query(description="是否在方言分类后运行 Whisper 转写"),
    ] = True,
    whisper_model: Annotated[
        Literal["tiny", "base", "small"],
        Query(description="语音转写所使用的 Whisper 模型"),
    ] = "base",
):
    suffix = Path(file.filename or "audio.wav").suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "unsupported_audio_format",
                "message": f"Supported formats: {sorted(SUPPORTED_SUFFIXES)}",
            },
        )

    payload = await file.read()
    max_bytes = APP.max_upload_mb * 1024 * 1024
    if not payload:
        raise HTTPException(
            status_code=400,
            detail={"code": "empty_file", "message": "The file is empty."},
        )
    if len(payload) > max_bytes:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "file_too_large",
                "message": f"Maximum upload size is {APP.max_upload_mb} MB.",
            },
        )

    tmp_path = None
    try:
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(payload)
            tmp_path = tmp.name

        result = run_pipeline(
            tmp_path,
            classifier=classifier,
            whisper_model=whisper_model,
            run_asr=run_asr,
        )
        result["probabilities"] = {
            DISPLAY_NAMES.get(label, label): score
            for label, score in zip(DIALECTS, result.pop("prob"))
        }
        return result
    except AudioProcessingError as exc:
        raise HTTPException(
            status_code=400,
            detail={"code": "audio_processing_failed", "message": str(exc)},
        ) from exc
    except DialectAppError as exc:
        raise HTTPException(
            status_code=503,
            detail={"code": "model_unavailable", "message": str(exc)},
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected prediction API failure")
        raise HTTPException(
            status_code=500,
            detail={
                "code": "internal_error",
                "message": "Unexpected prediction failure.",
            },
        ) from exc
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)
