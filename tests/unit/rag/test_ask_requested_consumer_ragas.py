import json
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import pytest
from redis.asyncio import Redis

from ai_service.observability.schemas import RagasEvalPayload
from ai_service.observability.service import RagasEvalService
from ai_service.rag.application.agentic_ask_use_case import AgenticAskUseCase
from ai_service.rag.application.ask_use_case import AskUseCase
from ai_service.rag.application.filter.rag_content_validator import RagContentValidator
from ai_service.rag.application.filter.secret_pii_scanner import SecretPiiScanner
from ai_service.rag.application.get_session_use_case import GetSessionUseCase
from ai_service.rag.application.get_sessions_use_case import GetSessionsUseCase
from ai_service.rag.application.query_complexity_router import QueryComplexityRouter
from ai_service.rag.infrastructure.messaging.ask_requested_consumer import (
    AskRequestedConsumer,
    AskRequestedMessage,
)
from ai_service.rag.rag_composition import RagComposition
from ai_service.rag.repository import ConversationSessionRepository
from ai_service.rag.schemas import GuardrailVerdict, IterationBudget


def _make_composition(
    ragas_eval_service: RagasEvalService | None = None,
) -> tuple[RagComposition, MagicMock]:
    ask_use_case_mock = MagicMock(spec=AskUseCase)
    router_mock = MagicMock(spec=QueryComplexityRouter)
    router_mock.route.return_value = "simple"

    validator_mock = MagicMock(spec=RagContentValidator)
    validator_mock.inspect_input.return_value = GuardrailVerdict.allow()

    session_repo_mock = MagicMock(spec=ConversationSessionRepository)
    session_repo_mock.find_by_id = AsyncMock(return_value=None)
    session_repo_mock.persist = AsyncMock(return_value=None)
    session_repo_mock.update = AsyncMock(return_value=None)

    composition = RagComposition(
        ask_use_case=ask_use_case_mock,
        agentic_ask_use_case=MagicMock(spec=AgenticAskUseCase),
        query_complexity_router=router_mock,
        rag_validator=validator_mock,
        secret_pii_scanner=SecretPiiScanner(),
        get_session_use_case=MagicMock(spec=GetSessionUseCase),
        get_sessions_use_case=MagicMock(spec=GetSessionsUseCase),
        delete_session_use_case=MagicMock(),
        session_repo=session_repo_mock,
        budget=IterationBudget.of(3, 1000, 5000),
        confidence_threshold=0.8,
        hyde_max_query_words=5,
        guardrail_enabled=False,
        ragas_eval_service=ragas_eval_service,
    )
    return composition, ask_use_case_mock


@pytest.mark.asyncio
async def test_ask_requested_consumer_triggers_ragas_eval_on_success() -> None:
    ragas_mock = MagicMock(spec=RagasEvalService)
    ragas_mock.evaluate = AsyncMock()

    composition, ask_mock = _make_composition(ragas_eval_service=ragas_mock)

    async def fake_ask_stream(command) -> AsyncIterator[str]:
        sources = [
            {"documentId": "doc-1", "snippet": "첫 번째 근거 본문", "score": 0.9},
            {"documentId": "doc-2", "snippet": "두 번째 근거 본문", "score": 0.8},
        ]
        yield f"__SOURCES:{json.dumps(sources)}"
        yield f"__CONFIDENCE:{json.dumps({'confidence': 0.9, 'missing': []})}"
        yield "답변입니다."

    ask_mock.execute.side_effect = fake_ask_stream

    redis_mock = MagicMock(spec=Redis)
    redis_mock.xadd = AsyncMock()
    redis_mock.get = AsyncMock(return_value=None)

    consumer = AskRequestedConsumer("localhost:9092", redis_mock, composition)
    message = AskRequestedMessage(
        job_id="job-ragas-1",
        user_id="user-1",
        question="스펙 질문",
    )

    await consumer._process(message)

    ragas_mock.evaluate.assert_awaited_once()
    payload: RagasEvalPayload = ragas_mock.evaluate.await_args[0][0]
    assert payload.trace_id == "job-ragas-1"
    assert payload.question == "스펙 질문"
    assert payload.answer == "답변입니다."
    assert payload.contexts == ["첫 번째 근거 본문", "두 번째 근거 본문"]


@pytest.mark.asyncio
async def test_ask_requested_consumer_does_not_trigger_ragas_when_cancelled() -> None:
    ragas_mock = MagicMock(spec=RagasEvalService)
    ragas_mock.evaluate = AsyncMock()

    composition, ask_mock = _make_composition(ragas_eval_service=ragas_mock)

    async def fake_ask_stream(command) -> AsyncIterator[str]:
        yield "취소 직전 토큰"

    ask_mock.execute.side_effect = fake_ask_stream

    redis_mock = MagicMock(spec=Redis)
    redis_mock.xadd = AsyncMock()
    # 취소 감지되도록 설정
    redis_mock.get = AsyncMock(return_value="1")

    consumer = AskRequestedConsumer("localhost:9092", redis_mock, composition)
    message = AskRequestedMessage(
        job_id="job-ragas-cancelled",
        user_id="user-1",
        question="취소될 질문",
    )

    await consumer._process(message)

    ragas_mock.evaluate.assert_not_awaited()


@pytest.mark.asyncio
async def test_ask_requested_consumer_isolates_ragas_eval_exception() -> None:
    ragas_mock = MagicMock(spec=RagasEvalService)
    ragas_mock.evaluate = AsyncMock(side_effect=RuntimeError("RAGAS 서비스 장애"))

    composition, ask_mock = _make_composition(ragas_eval_service=ragas_mock)

    async def fake_ask_stream(command) -> AsyncIterator[str]:
        yield "답변 내용입니다."

    ask_mock.execute.side_effect = fake_ask_stream

    redis_mock = MagicMock(spec=Redis)
    redis_mock.xadd = AsyncMock()
    redis_mock.get = AsyncMock(return_value=None)

    consumer = AskRequestedConsumer("localhost:9092", redis_mock, composition)
    message = AskRequestedMessage(
        job_id="job-ragas-fail",
        user_id="user-1",
        question="장애 격리 테스트 질문",
    )

    # 예외가 밖으로 전파되지 않고 완료되어야 함
    await consumer._process(message)

    ragas_mock.evaluate.assert_awaited_once()
    # done 이벤트가 정상 발행되었는지 확인
    done_calls = [
        call
        for call in redis_mock.xadd.await_args_list
        if call.args[1].get("type") == "done"
    ]
    assert len(done_calls) == 1


@pytest.mark.asyncio
async def test_ask_requested_consumer_does_not_trigger_ragas_when_no_collected_tokens() -> None:
    ragas_mock = MagicMock(spec=RagasEvalService)
    ragas_mock.evaluate = AsyncMock()

    composition, ask_mock = _make_composition(ragas_eval_service=ragas_mock)

    async def fake_ask_stream(command) -> AsyncIterator[str]:
        # 메타데이터 이벤트만 전달되고 실제 답변 토큰은 전혀 전달되지 않음
        yield f"__SOURCES:{json.dumps([])}"
        yield f"__CONFIDENCE:{json.dumps({'confidence': 0.5, 'missing': []})}"

    ask_mock.execute.side_effect = fake_ask_stream

    redis_mock = MagicMock(spec=Redis)
    redis_mock.xadd = AsyncMock()
    redis_mock.get = AsyncMock(return_value=None)

    consumer = AskRequestedConsumer("localhost:9092", redis_mock, composition)
    message = AskRequestedMessage(
        job_id="job-ragas-empty-tokens",
        user_id="user-1",
        question="빈 답변 테스트",
    )

    await consumer._process(message)

    ragas_mock.evaluate.assert_not_awaited()


@pytest.mark.asyncio
async def test_ask_requested_consumer_handles_none_ragas_service_gracefully() -> None:
    # ragas_eval_service가 주입되지 않은(None) 상태
    composition, ask_mock = _make_composition(ragas_eval_service=None)

    async def fake_ask_stream(command) -> AsyncIterator[str]:
        yield "정상적인 답변 토큰입니다."

    ask_mock.execute.side_effect = fake_ask_stream

    redis_mock = MagicMock(spec=Redis)
    redis_mock.xadd = AsyncMock()
    redis_mock.get = AsyncMock(return_value=None)

    consumer = AskRequestedConsumer("localhost:9092", redis_mock, composition)
    message = AskRequestedMessage(
        job_id="job-ragas-service-none",
        user_id="user-1",
        question="서비스 None 테스트",
    )

    # 예외 없이 정상 종료되어야 함
    await consumer._process(message)

    done_calls = [
        call
        for call in redis_mock.xadd.await_args_list
        if call.args[1].get("type") == "done"
    ]
    assert len(done_calls) == 1


@pytest.mark.asyncio
async def test_ask_requested_consumer_triggers_ragas_empty_contexts_when_no_sources(
) -> None:
    ragas_mock = MagicMock(spec=RagasEvalService)
    ragas_mock.evaluate = AsyncMock()

    composition, ask_mock = _make_composition(ragas_eval_service=ragas_mock)

    async def fake_ask_stream(command) -> AsyncIterator[str]:
        # sources 스트림 없이 답변만 생성된 경우
        yield "컨텍스트 없는 답변입니다."

    ask_mock.execute.side_effect = fake_ask_stream

    redis_mock = MagicMock(spec=Redis)
    redis_mock.xadd = AsyncMock()
    redis_mock.get = AsyncMock(return_value=None)

    consumer = AskRequestedConsumer("localhost:9092", redis_mock, composition)
    message = AskRequestedMessage(
        job_id="job-ragas-no-sources",
        user_id="user-1",
        question="소스 없는 질문",
    )

    await consumer._process(message)

    ragas_mock.evaluate.assert_awaited_once()
    payload: RagasEvalPayload = ragas_mock.evaluate.await_args[0][0]
    assert payload.trace_id == "job-ragas-no-sources"
    assert payload.question == "소스 없는 질문"
    assert payload.answer == "컨텍스트 없는 답변입니다."
    assert payload.contexts == []

