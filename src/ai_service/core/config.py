from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    node_env: str = "test"
    ai_service_port: int = 3004

    llm_provider: str = "ollama"
    embedding_provider: str = "ollama"

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:14b"
    ollama_embedding_model: str = "bge-m3"

    openai_api_key: str | None = None
    openai_model: str | None = None
    anthropic_api_key: str | None = None
    claude_model: str | None = None
    google_api_key: str | None = None
    google_model: str | None = None
    # text-embedding-004는 2026-01-14 폐기됨. gemini-embedding-001 사용 (GA, 무료 티어 제공)
    google_embedding_model: str = "gemini-embedding-001"
    groq_api_key: str | None = None
    groq_model: str | None = None

    llm_fallback_chain: str | None = None
    model_cost_table: str | None = None

    langsmith_tracing: bool = False
    langsmith_api_key: str | None = None
    langsmith_project: str = "ai-service"

    ragas_llm_eval_enabled: bool = False

    mongodb_vector_uri: str = "mongodb://localhost:27017"
    mongodb_db_name: str = "ai_service"

    redis_db_host: str = "localhost"
    redis_db_port: int = 6379

    kafka_brokers: str = "localhost:9092"

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None

    embedding_model: str | None = None
    embedding_dimension: int = 1536
    contextual_embeddings_enabled: bool = False

    admin_api_key: str | None = None

    hybrid_candidate_k: int | None = None
    reranker_top_n: int | None = None
    rrf_k: int | None = None
    reranker_enabled: bool = True
    reranker_api_url: str | None = None
    reranker_api_key: str | None = None

    llm_cache_ttl_seconds: int = 3600
    semantic_cache_enabled: bool = True
    semantic_cache_threshold: float = 0.85
    semantic_cache_ttl_seconds: int = 3600

    hyde_max_query_words: int = 5
    guardrail_enabled: bool = True

    agentic_max_iterations: int = 2
    agentic_token_budget: int = 30000
    agentic_timeout_ms: int = 20000
    agentic_confidence_threshold: float = 0.6


@lru_cache
def get_settings() -> Settings:
    return Settings()


__all__ = ["Settings", "get_settings"]
