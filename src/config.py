import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = PROJECT_ROOT / "outputs"
LOG_DIR = PROJECT_ROOT / "logs"


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 16000
    n_mels: int = 80
    n_fft: int = 400
    hop_length: int = 160
    win_length: int = 400
    max_seconds: float = 6.0
    window_seconds: float = 6.0
    hop_seconds: float = 3.0

    @property
    def max_samples(self):
        return int(self.sample_rate * self.max_seconds)


@dataclass(frozen=True)
class AppConfig:
    name: str = "中文地域方言识别与语音转写系统"
    version: str = "1.0.0"
    confidence_threshold: float = float(
        os.getenv("DIALECT_CONFIDENCE_THRESHOLD", "0.55")
    )
    default_classifier: str = os.getenv(
        "DIALECT_DEFAULT_CLASSIFIER",
        "whisper",
    )
    whisper_feature_model: str = os.getenv(
        "DIALECT_WHISPER_FEATURE_MODEL",
        "base",
    )
    max_upload_mb: int = int(os.getenv("DIALECT_MAX_UPLOAD_MB", "25"))


AUDIO = AudioConfig()
APP = AppConfig()

DIALECTS = ["shanghai", "changsha", "zhengzhou", "tianjin", "nanchang"]
DISPLAY_NAMES = {
    "shanghai": "上海话",
    "changsha": "长沙话",
    "zhengzhou": "郑州话",
    "tianjin": "天津话",
    "nanchang": "南昌话",
}

TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"
DID_CKPT = OUT_DIR / "did_cnn.pt"
WHISPER_CLASSIFIER_PATH = OUT_DIR / "whisper_classifier.joblib"
EXPERIMENT_RESULTS_PATH = OUT_DIR / "experiment_results.json"

# Backward-compatible names used by the feature and audio modules.
SR = AUDIO.sample_rate
N_MELS = AUDIO.n_mels
N_FFT = AUDIO.n_fft
HOP_LENGTH = AUDIO.hop_length
WIN_LENGTH = AUDIO.win_length
MAX_SEC = AUDIO.max_seconds
MAX_SAMPLES = AUDIO.max_samples
WINDOW_SEC = AUDIO.window_seconds
HOP_SEC = AUDIO.hop_seconds
CONFIDENCE_THRESHOLD = APP.confidence_threshold

for directory in (OUT_DIR, LOG_DIR):
    directory.mkdir(parents=True, exist_ok=True)
