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
async def test_consumer_complex_tracks_next_query_and_publishes_refined_queries_done():
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
                    "phase": "refining",
                    "confidence": 0.5,
                    "missing": ["부족"],
                    "nextQuery": "보완 질문 1",
                }
            )
            await command.on_progress(
                {
                    "iteration": 2,
                    "phase": "searching",
                    "confidence": 0.5,
                    "missing": ["부족"],
                    "nextQuery": "보완 질문 1",
                }
            )
            await command.on_progress(
                {
                    "iteration": 2,
                    "phase": "critiquing",
                    "confidence": 0.95,
                    "missing": [],
                    "refinedQueries": ["보완 질문 1"],
                }
            )
        yield "최종 답변입니다."

    agentic_use_case_mock.execute.side_effect = fake_agentic_execute

    router_mock = MagicMock(spec=QueryComplexityRouter)
    router_mock.route.return_value = "complex"

    validator_mock = MagicMock(spec=RagContentValidator)
    validator_mock.inspect_input.return_value = GuardrailVerdict.allow()

    initial_session = ConversationSession.create("user-1", "복잡한 에이전틱 질의")
    session_repo_mock = MagicMock(spec=ConversationSessionRepository)
    session_repo_mock.find_by_id = AsyncMock(return_value=initial_session)
    session_repo_mock.persist = AsyncMock(return_value=initial_session)
    session_repo_mock.update = AsyncMock()

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
        job_id="test-job-refined",
        user_id="user-1",
        session_id="session-123",
        question="복잡한 에이전틱 질의",
    )
    await consumer._process(message)

    # 1. progress 이벤트 확인 (nextQuery 포함)
    progress_calls = [
        call
        for call in redis_mock.xadd.await_args_list
        if call.args[1].get("type") == "progress"
    ]
    refining_call = next(
        c for c in progress_calls if json.loads(c.args[1]["data"]).get("phase") == "refining"
    )
    assert json.loads(refining_call.args[1]["data"])["nextQuery"] == "보완 질문 1"

    # 2. done 이벤트 확인 (refinedQueries 포함)
    done_calls = [
        call for call in redis_mock.xadd.await_args_list if call.args[1].get("type") == "done"
    ]
    assert len(done_calls) == 1
    done_data = json.loads(done_calls[0].args[1]["data"])
    assert done_data["refinedQueries"] == ["보완 질문 1"]

    # 3. session update 확인 (refined_queries 영속화)
    assert session_repo_mock.update.called
    saved_session: ConversationSession = session_repo_mock.update.call_args[0][0]
    assistant_turn = saved_session.turns[-1]
    assert assistant_turn.refined_queries == ["보완 질문 1"]


@pytest.mark.asyncio
async def test_consumer_complex_single_iteration_publishes_empty_refined_queries():
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
                {"iteration": 1, "phase": "critiquing", "confidence": 0.9, "missing": []}
            )
        yield "한 번에 만족된 답변입니다."

    agentic_use_case_mock.execute.side_effect = fake_agentic_execute

    router_mock = MagicMock(spec=QueryComplexityRouter)
    router_mock.route.return_value = "complex"

    validator_mock = MagicMock(spec=RagContentValidator)
    validator_mock.inspect_input.return_value = GuardrailVerdict.allow()

    initial_session = ConversationSession.create("user-1", "1회차 만족 질의")
    session_repo_mock = MagicMock(spec=ConversationSessionRepository)
    session_repo_mock.find_by_id = AsyncMock(return_value=initial_session)
    session_repo_mock.persist = AsyncMock(return_value=initial_session)
    session_repo_mock.update = AsyncMock()

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
        job_id="test-job-single",
        user_id="user-1",
        session_id="session-123",
        question="1회차 만족 질의",
    )
    await consumer._process(message)

    done_calls = [
        call for call in redis_mock.xadd.await_args_list if call.args[1].get("type") == "done"
    ]
    assert len(done_calls) == 1
    done_data = json.loads(done_calls[0].args[1]["data"])
    assert done_data["refinedQueries"] == []

    saved_session: ConversationSession = session_repo_mock.update.call_args[0][0]
    assistant_turn = saved_session.turns[-1]
    assert assistant_turn.refined_queries == []
