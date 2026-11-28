import json
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import pytest
from redis.asyncio import Redis

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
from ai_service.rag.schemas import (
    ConversationSession,
    GuardrailVerdict,
    IterationBudget,
)


@pytest.mark.asyncio
async def test_ask_requested_consumer_complex_exhausted_budget_publishes_done_and_persists():
    # 에이전틱 질의 중 예산 소진 시:
    # 1. done 이벤트에 isBudgetExhausted: True, confidence: 0.0 포함
    # 2. 세션 턴에 is_budget_exhausted=True 영속화
    redis_mock = MagicMock(spec=Redis)
    redis_mock.xadd = AsyncMock()
    redis_mock.get = AsyncMock(return_value=None)

    agentic_use_case_mock = MagicMock(spec=AgenticAskUseCase)

    async def fake_agentic_execute(command) -> AsyncIterator[str]:
        if command.on_progress:
            await command.on_progress(
                {"iteration": 1, "phase": "searching", "confidence": 0.0, "missing": []}
            )
            await command.on_progress(
                {
                    "iteration": 1,
                    "phase": "generating",
                    "confidence": 0.0,
                    "missing": [],
                    "isBudgetExhausted": True,
                }
            )
        yield "답변 내용입니다."
        yield '__CONFIDENCE:{"confidence": 0.0, "missing": [], "isBudgetExhausted": true}'

    agentic_use_case_mock.execute.side_effect = fake_agentic_execute

    router_mock = MagicMock(spec=QueryComplexityRouter)
    router_mock.route.return_value = "complex"

    validator_mock = MagicMock(spec=RagContentValidator)
    validator_mock.inspect_input.return_value = GuardrailVerdict.allow()

    session = ConversationSession.create("user-1", "예산 소진 질문")
    session_repo_mock = MagicMock(spec=ConversationSessionRepository)
    session_repo_mock.find_by_id = AsyncMock(return_value=session)
    session_repo_mock.persist = AsyncMock(return_value=session)
    session_repo_mock.update = AsyncMock(return_value=session)

    composition = RagComposition(
        ask_use_case=MagicMock(spec=AskUseCase),
        agentic_ask_use_case=agentic_use_case_mock,
        query_complexity_router=router_mock,
        rag_validator=validator_mock,
        secret_pii_scanner=SecretPiiScanner(),
        get_session_use_case=MagicMock(spec=GetSessionUseCase),
        get_sessions_use_case=MagicMock(spec=GetSessionsUseCase),
        delete_session_use_case=MagicMock(),
        session_repo=session_repo_mock,
        budget=IterationBudget.of(1, 1000, 5000),
        confidence_threshold=0.8,
        hyde_max_query_words=5,
        guardrail_enabled=False,
    )

    consumer = AskRequestedConsumer("localhost:9092", redis_mock, composition)

    message = AskRequestedMessage(
        job_id="job-exhausted-123",
        user_id="user-1",
        question="복잡한 질문",
        session_id=session.get_session_id(),
    )
    await consumer._process(message)

    # 1. done 이벤트 검증
    done_calls = [
        call for call in redis_mock.xadd.await_args_list if call.args[1].get("type") == "done"
    ]
    assert len(done_calls) == 1
    done_data = json.loads(done_calls[0].args[1]["data"])
    assert done_data["confidence"] == 0.0
    assert done_data["missing"] == []
    assert done_data["isBudgetExhausted"] is True

    # 2. session 영속화 확인
    session_repo_mock.update.assert_awaited_once()
    updated_session: ConversationSession = session_repo_mock.update.await_args.args[0]
    last_turn = updated_session.turns[-1]
    assert last_turn.role == "assistant"
    assert last_turn.is_budget_exhausted is True
    assert last_turn.confidence == 0.0


@pytest.mark.asyncio
async def test_ask_requested_consumer_complex_normal_publishes_is_budget_exhausted_false():
    # 정상 에이전틱 완료 시 done 이벤트에 isBudgetExhausted: False 포함
    redis_mock = MagicMock(spec=Redis)
    redis_mock.xadd = AsyncMock()
    redis_mock.get = AsyncMock(return_value=None)

    agentic_use_case_mock = MagicMock(spec=AgenticAskUseCase)

    async def fake_agentic_execute(command) -> AsyncIterator[str]:
        if command.on_progress:
            await command.on_progress(
                {"iteration": 1, "phase": "critiquing", "confidence": 0.9, "missing": []}
            )
        yield "정상 답변입니다."

    agentic_use_case_mock.execute.side_effect = fake_agentic_execute

    router_mock = MagicMock(spec=QueryComplexityRouter)
    router_mock.route.return_value = "complex"

    validator_mock = MagicMock(spec=RagContentValidator)
    validator_mock.inspect_input.return_value = GuardrailVerdict.allow()

    composition = RagComposition(
        ask_use_case=MagicMock(spec=AskUseCase),
        agentic_ask_use_case=agentic_use_case_mock,
        query_complexity_router=router_mock,
        rag_validator=validator_mock,
        secret_pii_scanner=SecretPiiScanner(),
        get_session_use_case=MagicMock(spec=GetSessionUseCase),
        get_sessions_use_case=MagicMock(spec=GetSessionsUseCase),
        delete_session_use_case=MagicMock(),
        session_repo=MagicMock(spec=ConversationSessionRepository),
        budget=IterationBudget.of(1, 1000, 5000),
        confidence_threshold=0.8,
        hyde_max_query_words=5,
        guardrail_enabled=False,
    )

    consumer = AskRequestedConsumer("localhost:9092", redis_mock, composition)

    message = AskRequestedMessage(
        job_id="job-normal-123",
        user_id="user-1",
        question="정상 질문",
    )
    await consumer._process(message)

    done_calls = [
        call for call in redis_mock.xadd.await_args_list if call.args[1].get("type") == "done"
    ]
    assert len(done_calls) == 1
    done_data = json.loads(done_calls[0].args[1]["data"])
    assert done_data["confidence"] == 0.9
    assert done_data["isBudgetExhausted"] is False
