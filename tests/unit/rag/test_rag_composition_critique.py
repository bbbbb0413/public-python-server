from contextlib import ExitStack
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from motor.motor_asyncio import AsyncIOMotorDatabase
from redis.asyncio import Redis

from ai_service.core.config import Settings
from ai_service.rag.application.critique_generator_service import CritiqueGeneratorService
from ai_service.rag.rag_composition import build_rag_composition


@pytest.mark.asyncio
async def test_build_rag_composition_injects_critique_generator_into_ask_use_case():
    settings = Settings(
        node_env="test",
        qdrant_url="http://localhost:6333",
        qdrant_api_key="key",
        embedding_dimension=1536,
    )
    mongo_db_mock = MagicMock(spec=AsyncIOMotorDatabase)
    mongo_db_mock.__getitem__.return_value = MagicMock()
    redis_mock = MagicMock(spec=Redis)
    redis_cb_mock = MagicMock(spec=Redis)

    patches = [
        "build_llm_provider",
        "build_embedding_provider",
        "CircuitBreakerAdapter",
        "LlmCostLogRepository",
        "CostTrackingService",
        "LlmRoutingService",
        "LangSmithTracingService",
        "LlmGatewayService",
        "PromptTemplateRepository",
        "PromptService",
        "AsyncQdrantClient",
        "QdrantVectorAdapter",
        "QdrantTextSearchAdapter",
        "HttpRerankerAdapter",
        "HydeService",
        "RrfFusionService",
        "QueryDecomposer",
        "HybridSearchUseCase",
        "RedisLlmCacheAdapter",
        "RedisSemanticCacheAdapter",
        "ConversationSessionRepository",
        "BookmarkRepository",
        "ConversationalQueryRewriter",
        "CritiqueGeneratorService",
        "QueryRefinerService",
        "AskUseCase",
        "AgenticAskUseCase",
        "RagasEvaluationRepository",
        "RagasEvalService",
    ]

    with ExitStack() as stack:
        mocks = {
            name: stack.enter_context(patch(f"ai_service.rag.rag_composition.{name}"))
            for name in patches
        }

        mock_vector_adapter = AsyncMock()
        mocks["QdrantVectorAdapter"].return_value = mock_vector_adapter

        mock_semantic_cache = AsyncMock()
        mocks["RedisSemanticCacheAdapter"].return_value = mock_semantic_cache

        mock_session_repo = AsyncMock()
        mocks["ConversationSessionRepository"].return_value = mock_session_repo

        mock_bookmark_repo = AsyncMock()
        mocks["BookmarkRepository"].return_value = mock_bookmark_repo

        sentinel_critique_generator = MagicMock(spec=CritiqueGeneratorService)
        mocks["CritiqueGeneratorService"].return_value = sentinel_critique_generator

        await build_rag_composition(settings, mongo_db_mock, redis_mock, redis_cb_mock)

        # AskUseCase 인스턴스화 시 critique_generator가 주입되었는지 검증
        mock_ask_use_case_cls = mocks["AskUseCase"]
        mock_ask_use_case_cls.assert_called_once()
        _, kwargs = mock_ask_use_case_cls.call_args
        assert kwargs.get("critique_generator") is sentinel_critique_generator
