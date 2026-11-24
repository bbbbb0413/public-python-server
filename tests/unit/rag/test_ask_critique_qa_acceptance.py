import json
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import pytest
from redis.asyncio import Redis

from ai_service.core.config import Settings
from ai_service.knowledge.schemas import (
    SimilaritySearchResult,
    VectorDocumentMetadata,
)
from ai_service.llm_gateway.application.llm_gateway_service import LlmGatewayService
from ai_service.prompt.schemas import PromptTemplate
from ai_service.prompt.service import PromptService
from ai_service.rag.application.agentic_ask_use_case import AgenticAskUseCase
from ai_service.rag.application.ask_use_case import AskUseCase
from ai_service.rag.application.conversational_query_rewriter_service import (
    ConversationalQueryRewriter,
)
from ai_service.rag.application.critique_generator_service import CritiqueGeneratorService
from ai_service.rag.application.filter.rag_content_validator import RagContentValidator
from ai_service.rag.application.filter.secret_pii_scanner import SecretPiiScanner
from ai_service.rag.application.get_session_use_case import GetSessionUseCase
from ai_service.rag.application.get_sessions_use_case import GetSessionsUseCase
from ai_service.rag.application.hybrid_search_use_case import (
    HybridSearchResult,
    HybridSearchUseCase,
)
from ai_service.rag.application.query_complexity_router import QueryComplexityRouter
from ai_service.rag.infrastructure.messaging.ask_requested_consumer import (
    AskRequestedConsumer,
    AskRequestedMessage,
)
from ai_service.rag.rag_composition import RagComposition
from ai_service.rag.repository import ConversationSessionRepository
from ai_service.rag.schemas import (
    ConversationSession,
    Critique,
    IterationBudget,
)


def _make_prompt_template(name: str = "rag-qa-system") -> PromptTemplate:
    return PromptTemplate.create(
        name,
        "Context: {{context}}\nDate: {{currentDate}}",
        ["context", "currentDate"],
    )


@pytest.fixture
def qa_settings() -> Settings:
    return Settings(
        llm_cache_ttl_seconds=3600,
        semantic_cache_enabled=False,
        semantic_cache_threshold=0.85,
        semantic_cache_ttl_seconds=3600,
    )


def _make_mock_llm_cache() -> AsyncMock:
    mock = AsyncMock()
    mock.get.return_value = None
    mock.set_with_ttl.return_value = None
    return mock


def _make_mock_semantic_cache() -> AsyncMock:
    mock = AsyncMock()
    mock.find_similar.return_value = None
    mock.store.return_value = None
    return mock


@pytest.mark.asyncio
async def test_acceptance_criteria_1_grounded_high_confidence(qa_settings: Settings):
    """수용 기준 1 검증:

    Given 사용자가 단순 질의를 전송하여 유사도 0.9인 청크가 검색되고,
          생성 답변이 청크를 충실히 반영하여 비평 결과 confidence=0.85, missing=[] 산출 시
    When 답변 생성이 완료되어 done 이벤트가 전달되면
    Then 최종 confidence=0.85, missing=[]가 done 이벤트 및 대화 세션에 정상 저장된다.
    """
    chunks = [
        SimilaritySearchResult(
            text="AI 서비스 사용 가이드 본문 내용입니다.",
            score=0.9,
            metadata=VectorDocumentMetadata(
                document_id="doc-guide-1",
                file_name="guide.pdf",
                chunk_index=0,
            ),
        )
    ]

    mock_llm_gateway = MagicMock(spec=LlmGatewayService)

    async def fake_stream(_command) -> AsyncIterator[str]:
        for token in ["가이드에", " ", "따른", " ", "답변입니다."]:
            yield token

    mock_llm_gateway.stream.side_effect = fake_stream

    mock_hybrid_search = AsyncMock(spec=HybridSearchUseCase)
    mock_hybrid_search.execute.return_value = HybridSearchResult(
        chunks=chunks,
        query_embedding=[0.1, 0.2],
    )

    mock_prompt_service = AsyncMock(spec=PromptService)
    mock_prompt_service.execute.return_value = _make_prompt_template()

    mock_critique_generator = AsyncMock(spec=CritiqueGeneratorService)
    mock_critique_generator.generate.return_value = Critique.of(
        answered=True, missing=[], next_query="", confidence=0.85
    )

    session_repo = MagicMock(spec=ConversationSessionRepository)
    initial_session = ConversationSession.create("user-qa", "가이드 질문")
    session_repo.find_by_id = AsyncMock(return_value=initial_session)
    session_repo.update = AsyncMock(return_value=initial_session)

    ask_use_case = AskUseCase(
        llm_gateway=mock_llm_gateway,
        hybrid_search=mock_hybrid_search,
        get_active_prompt=mock_prompt_service,
        llm_cache=_make_mock_llm_cache(),
        semantic_cache=_make_mock_semantic_cache(),
        settings=qa_settings,
        rag_validator=RagContentValidator(),
        secret_pii_scanner=SecretPiiScanner(),
        session_repo=session_repo,
        query_rewriter=AsyncMock(spec=ConversationalQueryRewriter),
        critique_generator=mock_critique_generator,
    )

    redis_mock = MagicMock(spec=Redis)
    redis_mock.xadd = AsyncMock()
    redis_mock.get = AsyncMock(return_value=None)

    router_mock = MagicMock(spec=QueryComplexityRouter)
    router_mock.route.return_value = "simple"

    composition = RagComposition(
        ask_use_case=ask_use_case,
        agentic_ask_use_case=MagicMock(spec=AgenticAskUseCase),
        query_complexity_router=router_mock,
        rag_validator=RagContentValidator(),
        secret_pii_scanner=SecretPiiScanner(),
        get_session_use_case=MagicMock(spec=GetSessionUseCase),
        get_sessions_use_case=MagicMock(spec=GetSessionsUseCase),
        delete_session_use_case=MagicMock(),
        session_repo=session_repo,
        budget=IterationBudget.of(3, 1000, 5000),
        confidence_threshold=0.8,
        hyde_max_query_words=5,
        guardrail_enabled=False,
    )

    consumer = AskRequestedConsumer("localhost:9092", redis_mock, composition)

    message = AskRequestedMessage(
        job_id="qa-job-grounded-verified",
        user_id="user-qa",
        question="가이드 질문",
        session_id=initial_session.get_session_id(),
    )
    await consumer._process(message)

    # 1. Critique 비평 생성 호출 검증
    mock_critique_generator.generate.assert_awaited_once_with(
        "가이드 질문", "가이드에 따른 답변입니다.", chunks, "default"
    )

    # 2. done 이벤트 페이로드 검증
    done_calls = [
        call
        for call in redis_mock.xadd.await_args_list
        if call.args[1].get("type") == "done"
    ]
    assert len(done_calls) == 1
    done_data = json.loads(done_calls[0].args[1]["data"])
    assert done_data["confidence"] == 0.85
    assert done_data["missing"] == []

    # 3. 세션 저장 검증
    session_repo.update.assert_awaited_once()
    saved_turn = session_repo.update.await_args[0][0].turns[1]
    assert saved_turn.content == "가이드에 따른 답변입니다."
    assert saved_turn.confidence == 0.85
    assert saved_turn.missing == []


@pytest.mark.asyncio
async def test_acceptance_criteria_2_hallucination_low_confidence_and_missing(
    qa_settings: Settings,
):
    """수용 기준 2 검증:

    Given 청크 유사도 0.88이나 LLM 환각으로 비평 결과
          confidence=0.35, missing=["핵심 수치 정보 불일치"] 산출 시
    When 답변 생성이 완료되면
    Then done 이벤트 및 세션에 해당 신뢰도와 결여 항목이 온전히 반영된다.
    """
    chunks = [
        SimilaritySearchResult(
            text="통계 보고서: 작년 총 매출액은 100억원입니다.",
            score=0.88,
            metadata=VectorDocumentMetadata(
                document_id="doc-report-1",
                file_name="report.pdf",
                chunk_index=0,
            ),
        )
    ]

    mock_llm_gateway = MagicMock(spec=LlmGatewayService)

    async def fake_stream(_command) -> AsyncIterator[str]:
        for token in ["작년", " ", "총", " ", "매출액은", " ", "5000억원입니다."]:
            yield token

    mock_llm_gateway.stream.side_effect = fake_stream

    mock_hybrid_search = AsyncMock(spec=HybridSearchUseCase)
    mock_hybrid_search.execute.return_value = HybridSearchResult(
        chunks=chunks,
        query_embedding=[0.1, 0.2],
    )

    mock_prompt_service = AsyncMock(spec=PromptService)
    mock_prompt_service.execute.return_value = _make_prompt_template()

    # 환각 감지 비평 결과 모킹
    mock_critique_generator = AsyncMock(spec=CritiqueGeneratorService)
    mock_critique_generator.generate.return_value = Critique.of(
        answered=False,
        missing=["핵심 수치 정보 불일치"],
        next_query="정확한 매출액 재확인",
        confidence=0.35,
    )

    session_repo = MagicMock(spec=ConversationSessionRepository)
    initial_session = ConversationSession.create("user-qa", "매출액 질문")
    session_repo.find_by_id = AsyncMock(return_value=initial_session)
    session_repo.update = AsyncMock(return_value=initial_session)

    ask_use_case = AskUseCase(
        llm_gateway=mock_llm_gateway,
        hybrid_search=mock_hybrid_search,
        get_active_prompt=mock_prompt_service,
        llm_cache=_make_mock_llm_cache(),
        semantic_cache=_make_mock_semantic_cache(),
        settings=qa_settings,
        rag_validator=RagContentValidator(),
        secret_pii_scanner=SecretPiiScanner(),
        session_repo=session_repo,
        query_rewriter=AsyncMock(spec=ConversationalQueryRewriter),
        critique_generator=mock_critique_generator,
    )

    redis_mock = MagicMock(spec=Redis)
    redis_mock.xadd = AsyncMock()
    redis_mock.get = AsyncMock(return_value=None)

    router_mock = MagicMock(spec=QueryComplexityRouter)
    router_mock.route.return_value = "simple"

    composition = RagComposition(
        ask_use_case=ask_use_case,
        agentic_ask_use_case=MagicMock(spec=AgenticAskUseCase),
        query_complexity_router=router_mock,
        rag_validator=RagContentValidator(),
        secret_pii_scanner=SecretPiiScanner(),
        get_session_use_case=MagicMock(spec=GetSessionUseCase),
        get_sessions_use_case=MagicMock(spec=GetSessionsUseCase),
        delete_session_use_case=MagicMock(),
        session_repo=session_repo,
        budget=IterationBudget.of(3, 1000, 5000),
        confidence_threshold=0.8,
        hyde_max_query_words=5,
        guardrail_enabled=False,
    )

    consumer = AskRequestedConsumer("localhost:9092", redis_mock, composition)

    message = AskRequestedMessage(
        job_id="qa-job-hallucination",
        user_id="user-qa",
        question="매출액 질문",
        session_id=initial_session.get_session_id(),
    )
    await consumer._process(message)

    # done 이벤트 검증 (검증 완료 배지 조건 불충족: confidence 0.35 < 0.8, missing 존재)
    done_calls = [
        call
        for call in redis_mock.xadd.await_args_list
        if call.args[1].get("type") == "done"
    ]
    assert len(done_calls) == 1
    done_data = json.loads(done_calls[0].args[1]["data"])
    assert done_data["confidence"] == 0.35
    assert done_data["missing"] == ["핵심 수치 정보 불일치"]

    # 세션 턴 검증
    session_repo.update.assert_awaited_once()
    saved_turn = session_repo.update.await_args[0][0].turns[1]
    assert saved_turn.confidence == 0.35
    assert saved_turn.missing == ["핵심 수치 정보 불일치"]


@pytest.mark.asyncio
async def test_acceptance_criteria_3_boundary_no_chunks(qa_settings: Settings):
    """수용 기준 3 검증:

    Given 검색 청크가 0건일 때 (경계 케이스)
    When 답변 생성이 완료되면
    Then 비평 호출 없이 confidence=0.0, missing=[]가 반영된다.
    """
    mock_llm_gateway = MagicMock(spec=LlmGatewayService)

    async def fake_stream(_command) -> AsyncIterator[str]:
        yield "문서에 관련 정보가 없습니다."

    mock_llm_gateway.stream.side_effect = fake_stream

    mock_hybrid_search = AsyncMock(spec=HybridSearchUseCase)
    mock_hybrid_search.execute.return_value = HybridSearchResult(
        chunks=[],
        query_embedding=[0.1, 0.2],
    )

    mock_prompt_service = AsyncMock(spec=PromptService)
    mock_prompt_service.execute.return_value = _make_prompt_template()

    mock_critique_generator = AsyncMock(spec=CritiqueGeneratorService)

    session_repo = MagicMock(spec=ConversationSessionRepository)
    initial_session = ConversationSession.create("user-qa", "없는 정보 질문")
    session_repo.find_by_id = AsyncMock(return_value=initial_session)
    session_repo.update = AsyncMock(return_value=initial_session)

    ask_use_case = AskUseCase(
        llm_gateway=mock_llm_gateway,
        hybrid_search=mock_hybrid_search,
        get_active_prompt=mock_prompt_service,
        llm_cache=_make_mock_llm_cache(),
        semantic_cache=_make_mock_semantic_cache(),
        settings=qa_settings,
        rag_validator=RagContentValidator(),
        secret_pii_scanner=SecretPiiScanner(),
        session_repo=session_repo,
        query_rewriter=AsyncMock(spec=ConversationalQueryRewriter),
        critique_generator=mock_critique_generator,
    )

    redis_mock = MagicMock(spec=Redis)
    redis_mock.xadd = AsyncMock()
    redis_mock.get = AsyncMock(return_value=None)

    router_mock = MagicMock(spec=QueryComplexityRouter)
    router_mock.route.return_value = "simple"

    composition = RagComposition(
        ask_use_case=ask_use_case,
        agentic_ask_use_case=MagicMock(spec=AgenticAskUseCase),
        query_complexity_router=router_mock,
        rag_validator=RagContentValidator(),
        secret_pii_scanner=SecretPiiScanner(),
        get_session_use_case=MagicMock(spec=GetSessionUseCase),
        get_sessions_use_case=MagicMock(spec=GetSessionsUseCase),
        delete_session_use_case=MagicMock(),
        session_repo=session_repo,
        budget=IterationBudget.of(3, 1000, 5000),
        confidence_threshold=0.8,
        hyde_max_query_words=5,
        guardrail_enabled=False,
    )

    consumer = AskRequestedConsumer("localhost:9092", redis_mock, composition)

    message = AskRequestedMessage(
        job_id="qa-job-no-chunks",
        user_id="user-qa",
        question="없는 정보 질문",
        session_id=initial_session.get_session_id(),
    )
    await consumer._process(message)

    # 비평 호출 건너뜀 검증
    mock_critique_generator.generate.assert_not_called()

    # done 이벤트 검증
    done_calls = [
        call
        for call in redis_mock.xadd.await_args_list
        if call.args[1].get("type") == "done"
    ]
    assert len(done_calls) == 1
    done_data = json.loads(done_calls[0].args[1]["data"])
    assert done_data["confidence"] == 0.0
    assert done_data["missing"] == []

    # 세션 검증
    session_repo.update.assert_awaited_once()
    saved_turn = session_repo.update.await_args[0][0].turns[1]
    assert saved_turn.confidence == 0.0
    assert saved_turn.missing == []


@pytest.mark.asyncio
async def test_acceptance_criteria_4_critique_service_exception_fallback(
    qa_settings: Settings,
):
    """수용 기준 4 검증:

    Given 비평 생성 서비스 호출 중 예외나 타임아웃 발생 시
    When 단순 질의 생성이 마무리되면
    Then 500 에러 없이 안전 폴백(confidence=0.0, missing=[])이 적용된다.
    """
    chunks = [
        SimilaritySearchResult(
            text="문서 정보",
            score=0.92,
            metadata=VectorDocumentMetadata(
                document_id="doc-fail-1",
                file_name="info.pdf",
                chunk_index=0,
            ),
        )
    ]

    mock_llm_gateway = MagicMock(spec=LlmGatewayService)

    async def fake_stream(_command) -> AsyncIterator[str]:
        yield "정상 답변 생성 중"

    mock_llm_gateway.stream.side_effect = fake_stream

    mock_hybrid_search = AsyncMock(spec=HybridSearchUseCase)
    mock_hybrid_search.execute.return_value = HybridSearchResult(
        chunks=chunks,
        query_embedding=[0.1, 0.2],
    )

    mock_prompt_service = AsyncMock(spec=PromptService)
    mock_prompt_service.execute.return_value = _make_prompt_template()

    # 비평 서비스에서 예외 발생
    mock_critique_generator = AsyncMock(spec=CritiqueGeneratorService)
    mock_critique_generator.generate.side_effect = TimeoutError("Critique LLM timeout")

    session_repo = MagicMock(spec=ConversationSessionRepository)
    initial_session = ConversationSession.create("user-qa", "타임아웃 질문")
    session_repo.find_by_id = AsyncMock(return_value=initial_session)
    session_repo.update = AsyncMock(return_value=initial_session)

    ask_use_case = AskUseCase(
        llm_gateway=mock_llm_gateway,
        hybrid_search=mock_hybrid_search,
        get_active_prompt=mock_prompt_service,
        llm_cache=_make_mock_llm_cache(),
        semantic_cache=_make_mock_semantic_cache(),
        settings=qa_settings,
        rag_validator=RagContentValidator(),
        secret_pii_scanner=SecretPiiScanner(),
        session_repo=session_repo,
        query_rewriter=AsyncMock(spec=ConversationalQueryRewriter),
        critique_generator=mock_critique_generator,
    )

    redis_mock = MagicMock(spec=Redis)
    redis_mock.xadd = AsyncMock()
    redis_mock.get = AsyncMock(return_value=None)

    router_mock = MagicMock(spec=QueryComplexityRouter)
    router_mock.route.return_value = "simple"

    composition = RagComposition(
        ask_use_case=ask_use_case,
        agentic_ask_use_case=MagicMock(spec=AgenticAskUseCase),
        query_complexity_router=router_mock,
        rag_validator=RagContentValidator(),
        secret_pii_scanner=SecretPiiScanner(),
        get_session_use_case=MagicMock(spec=GetSessionUseCase),
        get_sessions_use_case=MagicMock(spec=GetSessionsUseCase),
        delete_session_use_case=MagicMock(),
        session_repo=session_repo,
        budget=IterationBudget.of(3, 1000, 5000),
        confidence_threshold=0.8,
        hyde_max_query_words=5,
        guardrail_enabled=False,
    )

    consumer = AskRequestedConsumer("localhost:9092", redis_mock, composition)

    message = AskRequestedMessage(
        job_id="qa-job-critique-exception",
        user_id="user-qa",
        question="타임아웃 질문",
        session_id=initial_session.get_session_id(),
    )
    await consumer._process(message)

    # done 이벤트 검증 (안전 폴백 0.0)
    done_calls = [
        call
        for call in redis_mock.xadd.await_args_list
        if call.args[1].get("type") == "done"
    ]
    assert len(done_calls) == 1
    done_data = json.loads(done_calls[0].args[1]["data"])
    assert done_data["confidence"] == 0.0
    assert done_data["missing"] == []

    # 세션 검증
    session_repo.update.assert_awaited_once()
    saved_turn = session_repo.update.await_args[0][0].turns[1]
    assert saved_turn.confidence == 0.0
    assert saved_turn.missing == []
