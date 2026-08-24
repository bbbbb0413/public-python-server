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
from ai_service.rag.domain.model.conversation_session import ConversationSession
from ai_service.rag.domain.repository.conversation_session_repository import (
    IConversationSessionRepository,
)
from ai_service.rag.domain.vo.guardrail_verdict import GuardrailVerdict
from ai_service.rag.domain.vo.iteration_budget import IterationBudget
from ai_service.rag.infrastructure.messaging.ask_requested_consumer import (
    AskRequestedConsumer,
    AskRequestedMessage,
)
from ai_service.rag.rag_composition import RagComposition


@pytest.mark.asyncio
async def test_ask_requested_consumer_complex_publishes_progress():
    redis_mock = MagicMock(spec=Redis)
    redis_mock.xadd = AsyncMock()

    agentic_use_case_mock = MagicMock(spec=AgenticAskUseCase)

    async def fake_agentic_execute(command) -> AsyncIterator[str]:
        if command.on_progress:
            await command.on_progress(
                {"iteration": 1, "phase": "searching", "confidence": 0.0, "missing": []}
            )
            await command.on_progress(
                {"iteration": 1, "phase": "generating", "confidence": 0.0, "missing": []}
            )
            await command.on_progress(
                {"iteration": 1, "phase": "critiquing", "confidence": 0.9, "missing": []}
            )
        yield "답변입니다."

    agentic_use_case_mock.execute.side_effect = fake_agentic_execute

    router_mock = MagicMock(spec=QueryComplexityRouter)
    router_mock.route.return_value = "complex"

    validator_mock = MagicMock(spec=RagContentValidator)
    validator_mock.inspect_input.return_value = GuardrailVerdict.allow()

    session_repo_mock = MagicMock(spec=IConversationSessionRepository)
    session_repo_mock.find_by_id = AsyncMock(return_value=None)
    session_repo_mock.persist = AsyncMock(return_value=None)
    session_repo_mock.update = AsyncMock(return_value=None)

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
        budget=IterationBudget.of(3, 1000, 5000),
        confidence_threshold=0.8,
        hyde_max_query_words=5,
        guardrail_enabled=False,
    )

    consumer = AskRequestedConsumer("localhost:9092", redis_mock, composition)

    message = AskRequestedMessage(
        job_id="test-job-456",
        user_id="user-1",
        question="복잡한 에이전틱 질의",
    )
    await consumer._process(message)

    # redis xadd 호출 확인: progress 이벤트들이 발행되었는지
    progress_calls = [
        call
        for call in redis_mock.xadd.await_args_list
        if call.args[1].get("type") == "progress"
    ]
    assert len(progress_calls) == 3

    assert json.loads(progress_calls[0].args[1]["data"]) == {
        "iteration": 1,
        "phase": "searching",
        "confidence": 0.0,
        "missing": [],
    }
    assert json.loads(progress_calls[1].args[1]["data"]) == {
        "iteration": 1,
        "phase": "generating",
        "confidence": 0.0,
        "missing": [],
    }
    assert json.loads(progress_calls[2].args[1]["data"]) == {
        "iteration": 1,
        "phase": "critiquing",
        "confidence": 0.9,
        "missing": [],
    }


@pytest.mark.asyncio
async def test_ask_requested_consumer_simple_does_not_publish_progress():
    redis_mock = MagicMock(spec=Redis)
    redis_mock.xadd = AsyncMock()

    ask_use_case_mock = MagicMock(spec=AskUseCase)

    async def fake_ask_execute(_command) -> AsyncIterator[str]:
        yield "단순 답변입니다."

    ask_use_case_mock.execute.side_effect = fake_ask_execute

    router_mock = MagicMock(spec=QueryComplexityRouter)
    router_mock.route.return_value = "simple"

    validator_mock = MagicMock(spec=RagContentValidator)
    validator_mock.inspect_input.return_value = GuardrailVerdict.allow()

    session_repo_mock = MagicMock(spec=IConversationSessionRepository)
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
    )

    consumer = AskRequestedConsumer("localhost:9092", redis_mock, composition)

    message = AskRequestedMessage(
        job_id="test-job-789",
        user_id="user-1",
        question="단순 질의",
    )
    await consumer._process(message)

    # redis xadd 호출 확인: progress 이벤트가 전혀 발행되지 않아야 함
    progress_calls = [
        call
        for call in redis_mock.xadd.await_args_list
        if call.args[1].get("type") == "progress"
    ]
    assert len(progress_calls) == 0


@pytest.mark.asyncio
async def test_ask_requested_consumer_appends_sources_to_session():
    redis_mock = MagicMock(spec=Redis)
    redis_mock.xadd = AsyncMock()

    sources_data = [
        {"fileName": "test.pdf", "chunkIndex": 0, "documentId": "doc-1", "snippet": "테스트 요약"}
    ]

    ask_use_case_mock = MagicMock(spec=AskUseCase)

    async def fake_ask_execute(_command) -> AsyncIterator[str]:
        yield f"__SOURCES:{json.dumps(sources_data)}"
        yield "답변입니다."

    ask_use_case_mock.execute.side_effect = fake_ask_execute

    router_mock = MagicMock(spec=QueryComplexityRouter)
    router_mock.route.return_value = "simple"

    validator_mock = MagicMock(spec=RagContentValidator)
    validator_mock.inspect_input.return_value = GuardrailVerdict.allow()

    initial_session = ConversationSession.create("user-1", "질문입니다")
    session_repo_mock = MagicMock(spec=IConversationSessionRepository)
    session_repo_mock.find_by_id = AsyncMock(return_value=initial_session)
    session_repo_mock.persist = AsyncMock(return_value=initial_session)
    session_repo_mock.update = AsyncMock(return_value=initial_session)

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
    )

    consumer = AskRequestedConsumer("localhost:9092", redis_mock, composition)

    message = AskRequestedMessage(
        job_id="test-job-999",
        user_id="user-1",
        question="질문입니다",
        session_id=initial_session.get_session_id(),
    )
    await consumer._process(message)

    # session_repo.update 호출 인자 검증
    session_repo_mock.update.assert_awaited_once()
    updated_session = session_repo_mock.update.await_args[0][0]
    assert len(updated_session.turns) == 2
    assert updated_session.turns[1].role == "assistant"
    assert updated_session.turns[1].content == "답변입니다."
    assert updated_session.turns[1].sources == sources_data
