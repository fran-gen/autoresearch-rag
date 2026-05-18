from contextvars import ContextVar
from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


_runtime_google_api_key: ContextVar[str] = ContextVar(
    "runtime_google_api_key",
    default="",
)


class Settings(BaseSettings):
    """Application settings loaded from env vars and .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        populate_by_name=True,
        extra="ignore",
    )

    google_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("GOOGLE_API_KEY", "GEMINI_API_KEY"),
    )
    gemini_model: str = Field(default="gemini-2.5-pro", alias="GEMINI_MODEL")
    gemini_fast_model: str = Field(default="gemini-2.5-flash", alias="GEMINI_FAST_MODEL")
    embedding_model: str = Field(
        default="BAAI/bge-base-en-v1.5",
        alias="EMBEDDING_MODEL",
    )

    qdrant_path: Path = Field(default=Path("./index/qdrant_data"), alias="QDRANT_PATH")
    qdrant_url: str = Field(default="", alias="QDRANT_URL")

    benchmark_root: Path = Field(default=Path("./data"), alias="BENCHMARK_ROOT")
    experiment_db_path: Path = Field(
        default=Path("./data/experiments.db"),
        alias="EXPERIMENT_DB_PATH",
    )
    default_top_k: int = Field(default=8, alias="DEFAULT_TOP_K")
    model_inference_threads: int = Field(default=1, alias="MODEL_INFERENCE_THREADS")
    retrieve_top_k_cap: int = Field(default=24, alias="RETRIEVE_TOP_K_CAP")
    rerank_candidate_cap: int = Field(default=24, alias="RERANK_CANDIDATE_CAP")

    # Karpathy mode: when False (default), only `git add` pipeline.py after a run; when True, also commit.
    karpathy_pipeline_commit: bool = Field(default=False, alias="KARPATHY_PIPELINE_COMMIT")
    karpathy_sandbox_enabled: bool = Field(default=True, alias="KARPATHY_SANDBOX_ENABLED")
    karpathy_sandbox_fallback_to_process: bool = Field(
        default=True,
        alias="KARPATHY_SANDBOX_FALLBACK_TO_PROCESS",
    )
    karpathy_sandbox_image: str = Field(
        default="hackathon-lab-app:latest",
        alias="KARPATHY_SANDBOX_IMAGE",
    )
    karpathy_sandbox_timeout_seconds: int = Field(
        default=180,
        alias="KARPATHY_SANDBOX_TIMEOUT_SECONDS",
    )
    karpathy_sandbox_memory: str = Field(default="3g", alias="KARPATHY_SANDBOX_MEMORY")
    karpathy_sandbox_cpus: float = Field(default=1.0, alias="KARPATHY_SANDBOX_CPUS")
    karpathy_sandbox_threads: int = Field(default=1, alias="KARPATHY_SANDBOX_THREADS")
    karpathy_sandbox_network_disabled: bool = Field(
        default=True,
        alias="KARPATHY_SANDBOX_NETWORK_DISABLED",
    )
    karpathy_max_questions: int = Field(default=24, alias="KARPATHY_MAX_QUESTIONS")
    karpathy_retrieve_calls_per_question: int = Field(
        default=4,
        alias="KARPATHY_RETRIEVE_CALLS_PER_QUESTION",
    )
    karpathy_hf_cache_host_path: str = Field(
        default="",
        alias="KARPATHY_HF_CACHE_HOST_PATH",
    )
    karpathy_data_host_path: str = Field(
        default="",
        alias="KARPATHY_DATA_HOST_PATH",
    )
    karpathy_qdrant_host_path: str = Field(
        default="",
        alias="KARPATHY_QDRANT_HOST_PATH",
    )

    @property
    def has_google_key(self) -> bool:
        return bool(get_google_api_key())


def set_runtime_google_api_key(api_key: str) -> None:
    _runtime_google_api_key.set(api_key.strip())


def get_google_api_key() -> str:
    runtime_key = _runtime_google_api_key.get().strip()
    if runtime_key:
        return runtime_key
    return get_settings().google_api_key.strip()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
