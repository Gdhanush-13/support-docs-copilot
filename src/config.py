"""Configuration via environment variables with sensible defaults."""

from typing import List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # App
    APP_ENV: str = "development"
    SECRET_KEY: str = "change-me-in-production"
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8000"]

    # OpenAI
    OPENAI_API_KEY: str
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    OPENAI_CHAT_MODEL: str = "gpt-4o-mini"

    # Vector DB (Qdrant)
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_COLLECTION: str = "support_docs"
    QDRANT_API_KEY: str = ""

    # Redis (conversation memory)
    REDIS_URL: str = "redis://localhost:6379"
    MEMORY_TTL_SECONDS: int = 3600

    # PostgreSQL (metadata + eval logs)
    DATABASE_URL: str = "postgresql+asyncpg://copilot:copilot@localhost:5432/copilot"

    # Chunking
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64
    MAX_CHUNKS_PER_DOC: int = 500

    # Retrieval
    TOP_K_DENSE: int = 5
    TOP_K_SPARSE: int = 5
    HYBRID_ALPHA: float = 0.7  # weight for dense vs sparse

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
