import json
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient
from redis.asyncio import Redis

from ai_service.feedback.dependencies import get_answer_feedback_repository
from ai_service.feedback.repository import AnswerFeedbackRepository
from ai_service.knowledge.schemas import SimilaritySearchResult, VectorDocumentMetadata
from ai_service.llm_gateway.application.llm_gateway_service import LlmGatewayService
from ai_service.main import app
from ai_service.prompt.schemas import PromptTemplate
from ai_service.prompt.service import PromptService
from ai_service.rag.application.agentic_ask_use_case import (
    AgenticAskCommand,
    AgenticAskUseCase,
)
from ai_service.rag.application.ask_use_case import AskUseCase
from ai_service.rag.application.critique_generator_service import (
    Critique,
    CritiqueGeneratorService,
)
from ai_service.rag.application.filter.rag_content_validator import RagContentValidator
from ai_service.rag.application.filter.secret_pii_scanner import SecretPiiScanner
from ai_service.rag.application.get_session_use_case import GetSessionUseCase
from ai_service.rag.application.get_sessions_use_case import GetSessionsUseCase
from ai_service.rag.application.hybrid_search_use_case import (
    HybridSearchResult,
    HybridSearchUseCase,
)
from ai_service.rag.application.query_complexity_router import QueryComplexityRouter
from ai_service.rag.application.query_refiner_service import QueryRefinerService
from ai_service.rag.dependencies import (
    get_bookmark_repository,
    get_conversation_session_repository,
)
from ai_service.rag.infrastructure.messaging.ask_requested_consumer import (
    AskRequestedConsumer,
    AskRequestedMessage,
)
from ai_service.rag.rag_composition import RagComposition
from ai_service.rag.repository import BookmarkRepository, ConversationSessionRepository
from ai_service.rag.schemas import (
    ConversationSession,
    GuardrailVerdict,
    IterationBudget,
    RestoreProps,
    TurnRecord,
)


@pytest.fixture(autouse=True)
def _clear_dependencies_overrides():
    bookmark_repo_mock = MagicMock(spec=BookmarkRepository)
    bookmark_repo_mock.delete_by_session_for_user = AsyncMock(return_value=0)
    app.dependency_overrides[get_bookmark_repository] = lambda: bookmark_repo_mock

    feedback_repo_mock = MagicMock(spec=AnswerFeedbackRepository)
    feedback_repo_mock.delete_by_session_for_user = AsyncMock(return_value=0)
    app.dependency_overrides[get_answer_feedback_repository] = lambda: feedback_repo_mock
    yield
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_acceptance_criteria_1_first_iteration_budget_exhausted_done_event() -> None:
    """수용 기준 1 검증 (1회차 예산 소진):
    Given 에이전틱 RAG 질의 실행 중 1회차에서 시간/토큰 예산이 소진되었을 때
    When 스트리밍이 종료되고 완료(done) 이벤트가 발행되면
    Then done 이벤트 페이로드에 isBudgetExhausted: true가 포함되고, confidence는 0.0으로 전달된다.
    """
    redis_mock = MagicMock(spec=Redis)
    redis_mock.xadd = AsyncMock()
    redis_mock.get = AsyncMock(return_value=None)

    chunks = [
        SimilaritySearchResult(
            text="청크 내용 1",
            score=0.92,
            metadata=VectorDocumentMetadata(
                document_id="doc-1", file_name="doc1.pdf", chunk_index=0
            ),
        )
    ]
    mock_hybrid_search = AsyncMock(spec=HybridSearchUseCase)
    mock_hybrid_search.execute.return_value = HybridSearchResult(
        chunks=chunks, query_embedding=[0.1, 0.2]
    )

    mock_llm_gateway = MagicMock(spec=LlmGatewayService)

    async def fake_stream(_command):
        yield "예산 소진 직전 생성된 불완전한 답변 텍스트입니다."

    mock_llm_gateway.stream.side_effect = fake_stream

    mock_prompt_service = AsyncMock(spec=PromptService)
    mock_prompt_service.execute.return_value = PromptTemplate.create(
        "rag-qa-system", "Context: {{context}}", ["context"]
    )

    mock_critique = AsyncMock(spec=CritiqueGeneratorService)
    mock_refiner = MagicMock(spec=QueryRefinerService)

    agentic_use_case = AgenticAskUseCase(
        hybrid_search=mock_hybrid_search,
        llm_gateway=mock_llm_gateway,
        get_active_prompt=mock_prompt_service,
        critique_generator=mock_critique,
        query_refiner=mock_refiner,
        rag_validator=RagContentValidator(),
        secret_pii_scanner=SecretPiiScanner(),
    )

    router_mock = MagicMock(spec=QueryComplexityRouter)
    router_mock.route.return_value = "complex"

    validator_mock = MagicMock(spec=RagContentValidator)
    validator_mock.inspect_input.return_value = GuardrailVerdict.allow()

    session = ConversationSession.create("user-qa-1", "1회차 예산 소진 질문")
    session_repo_mock = MagicMock(spec=ConversationSessionRepository)
    session_repo_mock.find_by_id = AsyncMock(return_value=session)
    session_repo_mock.update = AsyncMock(return_value=session)

    composition = RagComposition(
        ask_use_case=MagicMock(spec=AskUseCase),
        agentic_ask_use_case=agentic_use_case,
        query_complexity_router=router_mock,
        rag_validator=validator_mock,
        secret_pii_scanner=SecretPiiScanner(),
        get_session_use_case=MagicMock(spec=GetSessionUseCase),
        get_sessions_use_case=MagicMock(spec=GetSessionsUseCase),
        delete_session_use_case=MagicMock(),
        session_repo=session_repo_mock,
        budget=IterationBudget.of(max_iterations=1, token_budget=1, timeout_ms=5000),
        confidence_threshold=0.8,
        hyde_max_query_words=5,
        guardrail_enabled=False,
    )

    consumer = AskRequestedConsumer("localhost:9092", redis_mock, composition)

    message = AskRequestedMessage(
        job_id="job-qa-ac1-first-iter",
        user_id="user-qa-1",
        question="1회차 예산 소진 시 완료 이벤트 검증",
        session_id=session.get_session_id(),
    )
    await consumer._process(message)

    # 비평 단계가 생략되었는지 확인
    mock_critique.generate.assert_not_awaited()

    # done 이벤트 단언
    done_calls = [
        call for call in redis_mock.xadd.await_args_list if call.args[1].get("type") == "done"
    ]
    assert len(done_calls) == 1
    done_payload = json.loads(done_calls[0].args[1]["data"])
    assert done_payload["isBudgetExhausted"] is True
    assert done_payload["confidence"] == 0.0
    assert done_payload["missing"] == []

    # 세션 턴에 is_budget_exhausted=True로 영속화되었는지 확인
    session_repo_mock.update.assert_awaited_once()
    saved_session: ConversationSession = session_repo_mock.update.await_args.args[0]
    assistant_turn = saved_session.turns[-1]
    assert assistant_turn.is_budget_exhausted is True
    assert assistant_turn.confidence == 0.0


@pytest.mark.asyncio
async def test_acceptance_criteria_1_multi_iteration_budget_exhausted_done_event() -> None:
    """수용 기준 1 검증 (다회차 예산 소진):
    Given 에이전틱 RAG 2회차에서 예산 소진 시 (1회차 비평 신뢰도가 0.85였더라도)
    When 스트리밍이 종료되고 완료(done) 이벤트가 발행되면
    Then 1회차 신뢰도가 왜곡 유지되지 않고 confidence 0.0 및 isBudgetExhausted: true가 전달된다.
    """
    redis_mock = MagicMock(spec=Redis)
    redis_mock.xadd = AsyncMock()
    redis_mock.get = AsyncMock(return_value=None)

    chunks = [
        SimilaritySearchResult(
            text="청크 내용 2",
            score=0.88,
            metadata=VectorDocumentMetadata(
                document_id="doc-2", file_name="doc2.pdf", chunk_index=0
            ),
        )
    ]
    mock_hybrid_search = AsyncMock(spec=HybridSearchUseCase)
    mock_hybrid_search.execute.return_value = HybridSearchResult(
        chunks=chunks, query_embedding=[0.1, 0.2]
    )

    mock_llm_gateway = MagicMock(spec=LlmGatewayService)

    async def fake_stream(_command):
        yield "2회차 답변 텍스트입니다."

    mock_llm_gateway.stream.side_effect = fake_stream

    mock_prompt_service = AsyncMock(spec=PromptService)
    mock_prompt_service.execute.return_value = PromptTemplate.create(
        "rag-qa-system", "Context: {{context}}", ["context"]
    )

    mock_critique = AsyncMock(spec=CritiqueGeneratorService)
    # 1회차 비평 점수가 0.85로 높게 나왔지만 기준(0.9) 미달로 2회차로 진행
    mock_critique.generate.return_value = Critique.of(
        answered=False,
        missing=["추가 보완 요망"],
        next_query="보완 검색어",
        confidence=0.85,
    )

    mock_refiner = MagicMock(spec=QueryRefinerService)
    mock_refiner.refine.return_value = "보완 검색어"

    agentic_use_case = AgenticAskUseCase(
        hybrid_search=mock_hybrid_search,
        llm_gateway=mock_llm_gateway,
        get_active_prompt=mock_prompt_service,
        critique_generator=mock_critique,
        query_refiner=mock_refiner,
        rag_validator=RagContentValidator(),
        secret_pii_scanner=SecretPiiScanner(),
    )

    router_mock = MagicMock(spec=QueryComplexityRouter)
    router_mock.route.return_value = "complex"

    validator_mock = MagicMock(spec=RagContentValidator)
    validator_mock.inspect_input.return_value = GuardrailVerdict.allow()

    session = ConversationSession.create("user-qa-1", "다회차 예산 소진 질문")
    session_repo_mock = MagicMock(spec=ConversationSessionRepository)
    session_repo_mock.find_by_id = AsyncMock(return_value=session)
    session_repo_mock.update = AsyncMock(return_value=session)

    # max_iterations=2로 설정하여 2회차 생성 후 예산 소진 유도
    composition = RagComposition(
        ask_use_case=MagicMock(spec=AskUseCase),
        agentic_ask_use_case=agentic_use_case,
        query_complexity_router=router_mock,
        rag_validator=validator_mock,
        secret_pii_scanner=SecretPiiScanner(),
        get_session_use_case=MagicMock(spec=GetSessionUseCase),
        get_sessions_use_case=MagicMock(spec=GetSessionsUseCase),
        delete_session_use_case=MagicMock(),
        session_repo=session_repo_mock,
        budget=IterationBudget.of(max_iterations=2, token_budget=10000, timeout_ms=5000),
        confidence_threshold=0.9,
        hyde_max_query_words=5,
        guardrail_enabled=False,
    )

    consumer = AskRequestedConsumer("localhost:9092", redis_mock, composition)

    message = AskRequestedMessage(
        job_id="job-qa-ac1-multi-iter",
        user_id="user-qa-1",
        question="다회차 예산 소진 시 신뢰도 정규화 검증",
        session_id=session.get_session_id(),
    )
    await consumer._process(message)

    # 1회차 비평은 1회만 호출되고 2회차 비평은 예산 소진으로 호출되지 않음
    assert mock_critique.generate.await_count == 1

    # done 이벤트 단언: 1회차 점수인 0.85가 아니라 0.0이어야 함
    done_calls = [
        call for call in redis_mock.xadd.await_args_list if call.args[1].get("type") == "done"
    ]
    assert len(done_calls) == 1
    done_payload = json.loads(done_calls[0].args[1]["data"])
    assert done_payload["isBudgetExhausted"] is True
    assert done_payload["confidence"] == 0.0
    assert done_payload["missing"] == []

    # 세션 턴 검증
    saved_session: ConversationSession = session_repo_mock.update.await_args.args[0]
    assistant_turn = saved_session.turns[-1]
    assert assistant_turn.is_budget_exhausted is True
    assert assistant_turn.confidence == 0.0


@pytest.mark.asyncio
async def test_acceptance_criteria_4_restored_session_turn_has_is_budget_exhausted_true() -> None:
    """수용 기준 4 검증 (세션 복원 API):
    Given 세션에 isBudgetExhausted: true 속성을 가진 답변 턴이 영속화되어 있을 때
    When 사용자가 세션 상세 조회 API(GET /rag/sessions/{sessionId}?userId=...)를 호출하면
    Then 응답 JSON의 turns 배열에 isBudgetExhausted: true가 유지되어 반환된다.
    """
    now = datetime.now(UTC)
    props = RestoreProps(
        session_id="session-qa-ac4",
        user_id="user-qa-4",
        title="예산 소진 세션 복원",
        turns=[
            TurnRecord(role="user", content="복잡한 에이전틱 질문", created_at=now),
            TurnRecord(
                role="assistant",
                content="비평이 생략된 답변 내용입니다.",
                created_at=now,
                confidence=0.0,
                missing=[],
                iterations=1,
                is_budget_exhausted=True,
            ),
        ],
        created_at=now,
        updated_at=now,
    )
    session = ConversationSession.restore(props)

    # 저장소 직렬화 및 역직렬화 무결성 확인
    record = ConversationSessionRepository._to_record(session)
    assert record["turns"][1]["isBudgetExhausted"] is True

    restored = ConversationSessionRepository._to_domain(record)
    assert restored.turns[1].is_budget_exhausted is True

    # E2E API 라우트 검증
    repo_mock = MagicMock(spec=ConversationSessionRepository)
    repo_mock.find_by_id_for_user = AsyncMock(return_value=session)
    app.dependency_overrides[get_conversation_session_repository] = lambda: repo_mock

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/rag/sessions/session-qa-ac4?userId=user-qa-4")

    assert response.status_code == 200
    body = response.json()
    assert body["sessionId"] == "session-qa-ac4"
    assert len(body["turns"]) == 2

    assistant_turn_json = body["turns"][1]
    assert assistant_turn_json["role"] == "assistant"
    assert assistant_turn_json["isBudgetExhausted"] is True
    assert assistant_turn_json["confidence"] == 0.0


@pytest.mark.asyncio
async def test_acceptance_criteria_5_legacy_session_and_normal_query_fallback_safe() -> None:
    """수용 기준 5 검증 (경계 케이스 및 레거시 폴백):
    Given [경계 케이스] isBudgetExhausted 필드가 없는 과거 레거시 또는 정상 완료 턴 조회 시
    When 세션 상세 조회 API를 호출하면
    Then isBudgetExhausted는 기본값 false로 안전하게 취급되어 어떠한 파싱 에러도 발생하지 않는다.
    """
    now = datetime.now(UTC)
    # 레거시 DB 도큐먼트 (isBudgetExhausted 필드 부재)
    legacy_mongo_record: dict[str, Any] = {
        "sessionId": "session-legacy-ac5",
        "userId": "user-qa-5",
        "title": "레거시 세션 데이터",
        "turns": [
            {"role": "user", "content": "단순 질의입니다.", "createdAt": now},
            {
                "role": "assistant",
                "content": "정상 단발성 응답입니다.",
                "createdAt": now,
                "confidence": 0.95,
                "missing": [],
                # isBudgetExhausted 필드 없음
            },
        ],
        "createdAt": now,
        "updatedAt": now,
    }

    # Repository 도메인 복원 검증
    session_from_legacy = ConversationSessionRepository._to_domain(legacy_mongo_record)
    assert session_from_legacy.turns[1].is_budget_exhausted is False

    # E2E API 라우트 응답 검증
    repo_mock = MagicMock(spec=ConversationSessionRepository)
    repo_mock.find_by_id_for_user = AsyncMock(return_value=session_from_legacy)
    app.dependency_overrides[get_conversation_session_repository] = lambda: repo_mock

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/rag/sessions/session-legacy-ac5?userId=user-qa-5")

    assert response.status_code == 200
    body = response.json()
    assert body["sessionId"] == "session-legacy-ac5"
    assert len(body["turns"]) == 2

    # 단발성 / 레거시 턴은 isBudgetExhausted가 false 여야 함
    assert body["turns"][0]["isBudgetExhausted"] is False
    assert body["turns"][1]["isBudgetExhausted"] is False
    assert body["turns"][1]["confidence"] == 0.95


@pytest.mark.asyncio
async def test_acceptance_criteria_1_mid_stream_token_exhaustion_in_agentic_loop() -> None:
    """수용 기준 1 추가 경계 검증 (생성 중 토큰 예산 초과로 조기 탈출):
    Given LLM 스트림 생성 도중 청크 누적 토큰이 예산을 초과하여 루프가 조기 중단될 때
    When AgenticAskUseCase가 실행되면
    Then 비평 단계를 건너뛰고 isBudgetExhausted: True 메타데이터 및 confidence 0.0을 방출한다.
    """
    chunks = [
        SimilaritySearchResult(
            text="청크 내용",
            score=0.99,
            metadata=VectorDocumentMetadata(
                document_id="doc-9", file_name="doc9.pdf", chunk_index=0
            ),
        )
    ]
    mock_hybrid_search = AsyncMock(spec=HybridSearchUseCase)
    mock_hybrid_search.execute.return_value = HybridSearchResult(
        chunks=chunks, query_embedding=[0.1, 0.2]
    )

    mock_llm_gateway = MagicMock(spec=LlmGatewayService)

    async def fake_stream_generating_many_tokens(_command):
        # 긴 토큰 청크를 여러 번 방출하여 생성 중 토큰 소진 유도
        for i in range(10):
            yield f"토큰 청크 {i} 번 데이터입니다. 길이가 매우 길어 토큰을 대량 소비합니다. "

    mock_llm_gateway.stream.side_effect = fake_stream_generating_many_tokens

    mock_prompt_service = AsyncMock(spec=PromptService)
    mock_prompt_service.execute.return_value = PromptTemplate.create(
        "rag-qa-system", "Context: {{context}}", ["context"]
    )

    mock_critique = AsyncMock(spec=CritiqueGeneratorService)
    mock_refiner = MagicMock(spec=QueryRefinerService)

    progress_events: list[dict[str, Any]] = []

    async def on_progress_callback(data: dict[str, Any]) -> None:
        progress_events.append(data)

    use_case = AgenticAskUseCase(
        hybrid_search=mock_hybrid_search,
        llm_gateway=mock_llm_gateway,
        get_active_prompt=mock_prompt_service,
        critique_generator=mock_critique,
        query_refiner=mock_refiner,
        rag_validator=RagContentValidator(),
        secret_pii_scanner=SecretPiiScanner(),
        on_progress=on_progress_callback,
    )

    # 매우 작은 token_budget(5)으로 즉각 중단 유도
    command = AgenticAskCommand(
        question="토큰 스트리밍 중 예산 소진 테스트",
        budget=IterationBudget.of(max_iterations=3, token_budget=5, timeout_ms=10000),
        confidence_threshold=0.8,
    )

    stream_chunks = [c async for c in use_case.execute(command)]

    # 비평 단계는 호출되지 않아야 함
    mock_critique.generate.assert_not_awaited()

    # 스트림 마지막 청크 확인
    confidence_chunks = [c for c in stream_chunks if c.startswith("__CONFIDENCE:")]
    assert len(confidence_chunks) == 1
    meta = json.loads(confidence_chunks[0][len("__CONFIDENCE:") :])
    assert meta["isBudgetExhausted"] is True
    assert meta["confidence"] == 0.0

    # 마지막 progress 이벤트 확인
    final_event = progress_events[-1]
    assert final_event["isBudgetExhausted"] is True
    assert final_event["confidence"] == 0.0
