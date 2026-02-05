from pydantic_settings import BaseSettings
from pydantic import field_validator
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def read_secret_file(file_path: str) -> Optional[str]:
    """Read a secret from a file, returning None if not found."""
    path = Path(file_path)
    if path.exists() and path.is_file():
        try:
            return path.read_text().strip()
        except Exception as e:
            logger.warning(f"Failed to read secret from {file_path}: {e}")
    return None


class Settings(BaseSettings):
    """Application configuration with environment variable support."""

    # Service settings
    bind_address: str = "0.0.0.0:8080"
    metrics_address: str = "0.0.0.0:9102"

    # Public API settings (GraphQL) - used for chat endpoint
    public_api_uri: str = "http://public-api:8080/api"
    public_api_timeout: int = 10

    # Product Service settings (REST API) - used for indexing
    product_service_uri: str = "http://product-api:9090"
    product_service_timeout: int = 30

    # Secrets directory (for file-mounted secrets)
    secrets_dir: str = "/etc/secrets"

    # Anthropic settings
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-20250514"
    anthropic_max_tokens: int = 1024

    @field_validator("anthropic_api_key", mode="after")
    @classmethod
    def load_anthropic_key_from_file(cls, v: str, info) -> str:
        """Load API key from file if not set via env var."""
        if v:
            return v

        # Try to read from secrets file
        secrets_dir = info.data.get("secrets_dir", "/etc/secrets")
        secret_path = f"{secrets_dir}/ANTHROPIC_API_KEY"
        file_value = read_secret_file(secret_path)

        if file_value:
            logger.info(f"Loaded ANTHROPIC_API_KEY from {secret_path}")
            return file_value

        logger.warning("ANTHROPIC_API_KEY not set via env var or file mount")
        return v

    # Vector store settings
    chroma_persist_dir: str = "/app/data/chroma"
    embedding_model: str = "all-MiniLM-L6-v2"

    # RAG settings
    retrieval_top_k: int = 5
    retrieval_threshold: float = 0.3

    # Indexer settings
    index_interval_seconds: int = 300

    # OpenTelemetry settings
    otel_exporter_otlp_traces_endpoint: str = "http://localhost:4318/v1/traces"
    otel_service_name: str = "hashicups-rag-api"
    otel_service_version: str = "1.0.0"

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
