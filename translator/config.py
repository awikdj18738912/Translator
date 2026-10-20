from dataclasses import dataclass, field
import os
from pathlib import Path
from urllib.parse import urlparse


def flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).lower() in {"1", "true", "yes"}


@dataclass(frozen=True)
class ModelConfig:
    backend: str = "ollama"
    base_url: str = "http://127.0.0.1:11434"
    model: str = "qwen3:4b"
    api_key: str = field(default="", repr=False)

    def validate(self, allow_remote: bool) -> None:
        if self.backend not in {"ollama", "compatible"}:
            raise ValueError("Model backend must be ollama or compatible")
        url = urlparse(self.base_url)
        if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password:
            raise ValueError("Model URL must be an HTTP(S) endpoint without embedded credentials")
        if not allow_remote and url.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("Local mode rejects remote model URLs; set ALLOW_REMOTE_MODELS=1 explicitly")
        if not self.model.strip():
            raise ValueError("Model name cannot be empty")


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path("data")
    translation: ModelConfig = field(default_factory=ModelConfig)
    refiner: ModelConfig = field(default_factory=ModelConfig)
    refiner_protocol: str = "text"
    refiner_enabled: bool = True
    asr_model: str = "small"
    asr_device: str = "cpu"
    asr_compute_type: str = "int8"
    asr_threads: int = 4
    asr_local_only: bool = False
    max_audio_seconds: int = 600
    max_upload_mb: int = 50
    max_pending_jobs: int = 10
    chunk_chars: int = 800
    model_timeout_seconds: int = 600
    ollama_num_gpu: int = 0
    ollama_num_ctx: int = 4096
    allow_remote_models: bool = False

    @classmethod
    def from_env(cls):
        def model(prefix):
            return ModelConfig(
                backend=os.getenv(f"{prefix}_BACKEND", "ollama"),
                base_url=os.getenv(f"{prefix}_BASE_URL", "http://127.0.0.1:11434").rstrip("/"),
                model=os.getenv(f"{prefix}_MODEL", "qwen3:4b"),
                api_key=os.getenv(f"{prefix}_API_KEY", ""),
            )
        return cls(
            data_dir=Path(os.getenv("DATA_DIR", "data")),
            translation=model("TRANSLATION"), refiner=model("REFINER"),
            refiner_protocol=os.getenv("REFINER_PROTOCOL", "text"),
            refiner_enabled=flag("REFINER_ENABLED", "1"),
            asr_model=os.getenv("ASR_MODEL", "small"),
            asr_device=os.getenv("ASR_DEVICE", "cpu"),
            asr_compute_type=os.getenv("ASR_COMPUTE_TYPE", "int8"),
            asr_threads=int(os.getenv("ASR_THREADS", "4")),
            asr_local_only=flag("ASR_LOCAL_ONLY"),
            max_audio_seconds=int(os.getenv("MAX_AUDIO_SECONDS", "600")),
            max_upload_mb=int(os.getenv("MAX_UPLOAD_MB", "50")),
            max_pending_jobs=int(os.getenv("MAX_PENDING_JOBS", "10")),
            chunk_chars=int(os.getenv("CHUNK_CHARS", "800")),
            model_timeout_seconds=int(os.getenv("MODEL_TIMEOUT_SECONDS", "600")),
            ollama_num_gpu=int(os.getenv("OLLAMA_NUM_GPU", "0")),
            ollama_num_ctx=int(os.getenv("OLLAMA_NUM_CTX", "4096")),
            allow_remote_models=flag("ALLOW_REMOTE_MODELS"),
        )

    def validate(self):
        self.translation.validate(self.allow_remote_models)
        self.refiner.validate(self.allow_remote_models)
        if self.refiner_protocol not in {"text", "codex_patch"}:
            raise ValueError("REFINER_PROTOCOL must be text or codex_patch")
        for name in ("asr_threads", "max_audio_seconds", "max_upload_mb", "max_pending_jobs", "model_timeout_seconds"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 100 <= self.chunk_chars <= 1200:
            raise ValueError("CHUNK_CHARS must be between 100 and 1200")
        if self.ollama_num_ctx < 2048:
            raise ValueError("OLLAMA_NUM_CTX must be at least 2048")

