from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


# Project root
BASE_DIR = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    ollama_model: str = "llama3.2"
    ollama_host: str = "http://127.0.0.1:11434"

    embedding_model: str = "all-MiniLM-L6-v2"
    top_k: int = 2

    chroma_path: Path = BASE_DIR / "backend" / "data" / "vector_store"
    chroma_collection: str = "construction_safety"

    yolo_model_path: Path = (
        BASE_DIR
        / "runs"
        / "detect"
        / "runs"
        / "construction_yolo11n"
        / "weights"
        / "best.pt"
    )

    confidence_threshold: float = 0.30

    api_title: str = "ConstructionSafe AI API"
    api_version: str = "1.0.0"

    cors_origins: str = "http://localhost:8501"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


settings = Settings()