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
    redis_mock.get = AsyncMock(return_value=None)

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
    redis_mock.get = AsyncMock(return_value=None)

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

    # redis xadd 호출 확인: done 이벤트가 메타데이터 없이 발행되었는지
    done_calls = [
        call
        for call in redis_mock.xadd.await_args_list
        if call.args[1].get("type") == "done"
    ]
    assert len(done_calls) == 1
    assert "data" not in done_calls[0].args[1]


@pytest.mark.asyncio
async def test_ask_requested_consumer_appends_sources_to_session():
    redis_mock = MagicMock(spec=Redis)
    redis_mock.xadd = AsyncMock()
    redis_mock.get = AsyncMock(return_value=None)

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


@pytest.mark.asyncio
async def test_ask_requested_consumer_complex_publishes_done_with_metadata():
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
                {"iteration": 1, "phase": "generating", "confidence": 0.0, "missing": []}
            )
            await command.on_progress(
                {
                    "iteration": 1,
                    "phase": "critiquing",
                    "confidence": 0.85,
                    "missing": ["추가 설명"],
                }
            )
        yield "답변입니다."

    agentic_use_case_mock.execute.side_effect = fake_agentic_execute

    router_mock = MagicMock(spec=QueryComplexityRouter)
    router_mock.route.return_value = "complex"

    validator_mock = MagicMock(spec=RagContentValidator)
    validator_mock.inspect_input.return_value = GuardrailVerdict.allow()

    initial_session = ConversationSession.create("user-1", "질문입니다")
    session_repo_mock = MagicMock(spec=IConversationSessionRepository)
    session_repo_mock.find_by_id = AsyncMock(return_value=initial_session)
    session_repo_mock.persist = AsyncMock(return_value=initial_session)
    session_repo_mock.update = AsyncMock(return_value=initial_session)

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
        job_id="test-job-complex-done",
        user_id="user-1",
        question="복잡한 에이전틱 질의",
    )
    await consumer._process(message)

    # redis xadd 호출 확인: done 이벤트에 confidence, missing 메타데이터가 포함되었는지
    done_calls = [
        call
        for call in redis_mock.xadd.await_args_list
        if call.args[1].get("type") == "done"
    ]
    assert len(done_calls) == 1
    assert "data" in done_calls[0].args[1]
    done_data = json.loads(done_calls[0].args[1]["data"])
    assert done_data == {"confidence": 0.85, "missing": ["추가 설명"]}


@pytest.mark.asyncio
async def test_ask_requested_consumer_cancelled_stops_token_streaming():
    redis_mock = MagicMock(spec=Redis)
    redis_mock.xadd = AsyncMock()

    # 첫 토큰 이후 취소 플래그가 설정된 상황을 모사
    # get(f"job:{job_id}:cancelled") 호출 시 1회차는 None, 2회차는 b"1" (또는 "true")
    redis_mock.get = AsyncMock(side_effect=[None, b"1", b"1"])

    ask_use_case_mock = MagicMock(spec=AskUseCase)

    async def fake_ask_execute(_command) -> AsyncIterator[str]:
        yield "첫번째 토큰 "
        yield "두번째 토큰 "
        yield "세번째 토큰 "

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
        job_id="test-job-cancel-123",
        user_id="user-1",
        question="질문입니다",
        session_id=initial_session.get_session_id(),
    )
    await consumer._process(message)

    # 토큰 발행은 첫번째 토큰만 이루어져야 함
    token_calls = [
        call
        for call in redis_mock.xadd.await_args_list
        if call.args[1].get("type") == "token"
    ]
    assert len(token_calls) == 1
    assert token_calls[0].args[1]["data"] == "첫번째 토큰 "

    # 세션 저장은 중단 시점까지 수집된 "첫번째 토큰 "으로 완료되어야 함
    session_repo_mock.update.assert_awaited_once()
    updated_session = session_repo_mock.update.await_args[0][0]
    assert len(updated_session.turns) == 2
    assert updated_session.turns[1].content == "첫번째 토큰 "

    # done 이벤트가 정상 발행되어야 함
    done_calls = [
        call
        for call in redis_mock.xadd.await_args_list
        if call.args[1].get("type") == "done"
    ]
    assert len(done_calls) == 1


@pytest.mark.asyncio
async def test_ask_requested_consumer_cancelled_before_first_chunk():
    redis_mock = MagicMock(spec=Redis)
    redis_mock.xadd = AsyncMock()

    # 스트림 진입 전부터 이미 취소 플래그가 설정되어 있는 상황
    redis_mock.get = AsyncMock(return_value=b"1")

    ask_use_case_mock = MagicMock(spec=AskUseCase)

    async def fake_ask_execute(_command) -> AsyncIterator[str]:
        yield "첫번째 토큰 "
        yield "두번째 토큰 "

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
        job_id="test-job-cancel-immediate",
        user_id="user-1",
        question="질문입니다",
        session_id=initial_session.get_session_id(),
    )
    await consumer._process(message)

    # 토큰 발행이 전혀 되지 않아야 함
    token_calls = [
        call
        for call in redis_mock.xadd.await_args_list
        if call.args[1].get("type") == "token"
    ]
    assert len(token_calls) == 0

    # 수집된 토큰이 없으므로 session_repo.update가 호출되지 않음
    session_repo_mock.update.assert_not_called()

    # done 이벤트는 발행되어야 함
    done_calls = [
        call
        for call in redis_mock.xadd.await_args_list
        if call.args[1].get("type") == "done"
    ]
    assert len(done_calls) == 1




