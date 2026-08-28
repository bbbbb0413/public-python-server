import logging
from dataclasses import dataclass

from motor.motor_asyncio import AsyncIOMotorDatabase
from qdrant_client import AsyncQdrantClient
from redis.asyncio import Redis

from ai_service.core.config import Settings
from ai_service.knowledge.infrastructure.providers.embedding_factory import build_embedding_provider
from ai_service.knowledge.repository import QdrantVectorAdapter
from ai_service.llm_gateway.application.cost_tracking_service import (
    CostTrackingService,
    parse_cost_table,
)
from ai_service.llm_gateway.application.fallback_service import FallbackService
from ai_service.llm_gateway.application.langsmith_tracing_service import LangSmithTracingService
from ai_service.llm_gateway.application.llm_gateway_service import LlmGatewayService
from ai_service.llm_gateway.application.llm_routing_service import LlmRoutingService
from ai_service.llm_gateway.circuit_breaker import CircuitBreakerAdapter
from ai_service.llm_gateway.infrastructure.providers.factory import build_llm_provider
from ai_service.llm_gateway.repository import LlmCostLogRepository
from ai_service.prompt.repository import PromptTemplateRepository
from ai_service.prompt.service import PromptService
from ai_service.rag.application.agentic_ask_use_case import AgenticAskUseCase
from ai_service.rag.application.ask_use_case import AskUseCase
from ai_service.rag.application.conversational_query_rewriter_service import (
    ConversationalQueryRewriter,
)
from ai_service.rag.application.critique_generator_service import CritiqueGeneratorService
from ai_service.rag.application.delete_session_use_case import DeleteSessionUseCase
from ai_service.rag.application.filter.rag_content_validator import RagContentValidator
from ai_service.rag.application.filter.secret_pii_scanner import SecretPiiScanner
from ai_service.rag.application.get_session_use_case import GetSessionUseCase
from ai_service.rag.application.get_sessions_use_case import GetSessionsUseCase
from ai_service.rag.application.hybrid_search_use_case import HybridSearchUseCase
from ai_service.rag.application.hyde_service import HydeService
from ai_service.rag.application.query_complexity_router import QueryComplexityRouter
from ai_service.rag.application.query_decomposer_service import QueryDecomposer
from ai_service.rag.application.query_refiner_service import QueryRefinerService
from ai_service.rag.application.rrf_fusion_service import RrfFusionService
from ai_service.rag.infrastructure.cache.redis_llm_cache_adapter import RedisLlmCacheAdapter
from ai_service.rag.infrastructure.cache.redis_semantic_cache_adapter import (
    RedisSemanticCacheAdapter,
)
from ai_service.rag.infrastructure.search.http_reranker_adapter import HttpRerankerAdapter
from ai_service.rag.infrastructure.search.qdrant_text_search_adapter import QdrantTextSearchAdapter
from ai_service.rag.repository import ConversationSessionRepository
from ai_service.rag.schemas import IterationBudget

LLM_CACHE_DB = 2

logger = logging.getLogger(__name__)


@dataclass
class RagComposition:
    ask_use_case: AskUseCase
    agentic_ask_use_case: AgenticAskUseCase
    query_complexity_router: QueryComplexityRouter
    rag_validator: RagContentValidator
    secret_pii_scanner: SecretPiiScanner
    get_session_use_case: GetSessionUseCase
    get_sessions_use_case: GetSessionsUseCase
    delete_session_use_case: DeleteSessionUseCase
    session_repo: ConversationSessionRepository
    budget: IterationBudget
    confidence_threshold: float
    hyde_max_query_words: int
    guardrail_enabled: bool


async def build_rag_composition(
    settings: Settings,
    mongo_db: AsyncIOMotorDatabase[dict[str, object]],
    redis_client: Redis,
    redis_cb_client: Redis,
) -> RagComposition:
    llm_provider = build_llm_provider(settings)
    embedding_provider = build_embedding_provider(settings)

    breaker = CircuitBreakerAdapter(redis_cb_client)
    fallback = FallbackService(llm_provider, breaker)
    cost_repo = LlmCostLogRepository(mongo_db)
    cost_tracking = CostTrackingService(cost_repo, parse_cost_table(settings.model_cost_table))
    routing = LlmRoutingService(settings)
    langsmith = LangSmithTracingService(settings)
    llm_gateway_service = LlmGatewayService(fallback, cost_tracking, routing, langsmith)

    prompt_repo = PromptTemplateRepository(mongo_db)
    prompt_service = PromptService(prompt_repo)

    qdrant_client = AsyncQdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    vector_store = QdrantVectorAdapter(qdrant_client, settings.embedding_dimension)
    await vector_store.ensure_collection()
    lexical_search = QdrantTextSearchAdapter(qdrant_client)

    reranker = HttpRerankerAdapter(settings.reranker_api_url, settings.reranker_api_key)
    rrf_fusion = RrfFusionService()
    hyde_service = HydeService(llm_provider)
    query_decomposer = QueryDecomposer(llm_provider)

    hybrid_search = HybridSearchUseCase(
        embedding_provider,
        vector_store,
        lexical_search,
        reranker,
        rrf_fusion,
        hyde_service,
        query_decomposer,
        settings,
    )

    llm_cache_redis: Redis = Redis(
        host=settings.redis_db_host,
        port=settings.redis_db_port,
        db=LLM_CACHE_DB,
        decode_responses=True,
    )
    llm_cache = RedisLlmCacheAdapter(llm_cache_redis)

    semantic_cache_redis: Redis = Redis(
        host=settings.redis_db_host, port=settings.redis_db_port, decode_responses=False
    )
    semantic_cache = RedisSemanticCacheAdapter(semantic_cache_redis, settings.embedding_dimension)
    try:
        await semantic_cache.ensure_index()
    except Exception:
        logger.warning(
            "semantic cache 인덱스 생성 실패 - RediSearch 모듈이 없는 Redis로 추정됨. "
            "semantic cache 조회는 항상 미스로 동작합니다.",
            exc_info=True,
        )

    session_repo = ConversationSessionRepository(mongo_db)
    await session_repo.ensure_indexes()

    rag_validator = RagContentValidator()
    secret_pii_scanner = SecretPiiScanner()
    query_rewriter = ConversationalQueryRewriter(llm_provider)

    ask_use_case = AskUseCase(
        llm_gateway_service,
        hybrid_search,
        prompt_service,
        llm_cache,
        semantic_cache,
        settings,
        rag_validator,
        secret_pii_scanner,
        session_repo,
        query_rewriter,
    )

    critique_generator = CritiqueGeneratorService(llm_gateway_service, settings)
    query_refiner = QueryRefinerService()
    agentic_ask_use_case = AgenticAskUseCase(
        hybrid_search,
        llm_gateway_service,
        prompt_service,
        critique_generator,
        query_refiner,
        rag_validator,
        secret_pii_scanner,
    )

    budget = IterationBudget.of(
        settings.agentic_max_iterations,
        settings.agentic_token_budget,
        settings.agentic_timeout_ms,
    )

    return RagComposition(
        ask_use_case=ask_use_case,
        agentic_ask_use_case=agentic_ask_use_case,
        query_complexity_router=QueryComplexityRouter(),
        rag_validator=rag_validator,
        secret_pii_scanner=secret_pii_scanner,
        get_session_use_case=GetSessionUseCase(session_repo),
        get_sessions_use_case=GetSessionsUseCase(session_repo),
        delete_session_use_case=DeleteSessionUseCase(session_repo),
        session_repo=session_repo,
        budget=budget,
        confidence_threshold=settings.agentic_confidence_threshold,
        hyde_max_query_words=settings.hyde_max_query_words,
        guardrail_enabled=settings.guardrail_enabled,
    )
