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

    session_repo_mock = MagicMock(spec=ConversationSessionRepository)
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
async def test_ask_requested_consumer_simple_publishes_searching_and_generating_progress():
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
    )

    consumer = AskRequestedConsumer("localhost:9092", redis_mock, composition)

    message = AskRequestedMessage(
        job_id="test-job-789",
        user_id="user-1",
        question="단순 질의",
    )
    await consumer._process(message)

    # redis xadd 호출 확인: 단순 경로도 searching → generating 진행 이벤트를 발행해야 함
    progress_calls = [
        call
        for call in redis_mock.xadd.await_args_list
        if call.args[1].get("type") == "progress"
    ]
    assert len(progress_calls) == 2
    assert json.loads(progress_calls[0].args[1]["data"]) == {
        "iteration": 0,
        "phase": "searching",
    }
    assert json.loads(progress_calls[1].args[1]["data"]) == {
        "iteration": 0,
        "phase": "generating",
    }

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
    session_repo_mock = MagicMock(spec=ConversationSessionRepository)
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
    assert updated_session.turns[1].confidence is None
    assert updated_session.turns[1].missing is None


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
    session_repo_mock = MagicMock(spec=ConversationSessionRepository)
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

    # redis xadd 호출 확인: done 이벤트에 confidence, missing, iterations 메타데이터가 포함되었는지
    done_calls = [
        call
        for call in redis_mock.xadd.await_args_list
        if call.args[1].get("type") == "done"
    ]
    assert len(done_calls) == 1
    assert "data" in done_calls[0].args[1]
    done_data = json.loads(done_calls[0].args[1]["data"])
    assert done_data == {"confidence": 0.85, "missing": ["추가 설명"], "iterations": 1}


@pytest.mark.asyncio
async def test_ask_requested_consumer_complex_multiple_iterations_publishes_latest_iteration():
    redis_mock = MagicMock(spec=Redis)
    redis_mock.xadd = AsyncMock()
    redis_mock.get = AsyncMock(return_value=None)

    agentic_use_case_mock = MagicMock(spec=AgenticAskUseCase)

    async def fake_agentic_execute(command) -> AsyncIterator[str]:
        if command.on_progress:
            # 1회차 진행 및 비평 불만족
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
                    "confidence": 0.5,
                    "missing": ["상세 정보 필요"],
                }
            )
            await command.on_progress(
                {
                    "iteration": 1,
                    "phase": "refining",
                    "confidence": 0.5,
                    "missing": ["상세 정보 필요"],
                }
            )
            # 2회차 재검색 및 비평 만족
            await command.on_progress(
                {"iteration": 2, "phase": "searching", "confidence": 0.5, "missing": []}
            )
            await command.on_progress(
                {"iteration": 2, "phase": "generating", "confidence": 0.5, "missing": []}
            )
            await command.on_progress(
                {"iteration": 2, "phase": "critiquing", "confidence": 0.95, "missing": []}
            )
        yield "개선된 2회차 최종 답변입니다."

    agentic_use_case_mock.execute.side_effect = fake_agentic_execute

    router_mock = MagicMock(spec=QueryComplexityRouter)
    router_mock.route.return_value = "complex"

    validator_mock = MagicMock(spec=RagContentValidator)
    validator_mock.inspect_input.return_value = GuardrailVerdict.allow()

    initial_session = ConversationSession.create("user-1", "복합 질의")
    session_repo_mock = MagicMock(spec=ConversationSessionRepository)
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
        job_id="test-job-complex-multi-iteration",
        user_id="user-1",
        question="복합 질의",
    )
    await consumer._process(message)

    done_calls = [
        call
        for call in redis_mock.xadd.await_args_list
        if call.args[1].get("type") == "done"
    ]
    assert len(done_calls) == 1
    assert "data" in done_calls[0].args[1]
    done_data = json.loads(done_calls[0].args[1]["data"])
    assert done_data == {"confidence": 0.95, "missing": [], "iterations": 2}

    session_repo_mock.update.assert_awaited_once()
    updated_session = session_repo_mock.update.await_args[0][0]
    assert len(updated_session.turns) == 2
    assert updated_session.turns[0].iterations is None
    assert updated_session.turns[1].role == "assistant"
    assert updated_session.turns[1].content == "개선된 2회차 최종 답변입니다."
    assert updated_session.turns[1].confidence == 0.95
    assert updated_session.turns[1].missing == []
    assert updated_session.turns[1].iterations == 2


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
    session_repo_mock = MagicMock(spec=ConversationSessionRepository)
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
    session_repo_mock = MagicMock(spec=ConversationSessionRepository)
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


@pytest.mark.asyncio
async def test_ask_requested_consumer_complex_persists_confidence_and_missing_to_session():
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
        yield "에이전틱 답변입니다."

    agentic_use_case_mock.execute.side_effect = fake_agentic_execute

    router_mock = MagicMock(spec=QueryComplexityRouter)
    router_mock.route.return_value = "complex"

    validator_mock = MagicMock(spec=RagContentValidator)
    validator_mock.inspect_input.return_value = GuardrailVerdict.allow()

    initial_session = ConversationSession.create("user-1", "복잡한 에이전틱 질문")
    session_repo_mock = MagicMock(spec=ConversationSessionRepository)
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
        job_id="test-job-session-confidence",
        user_id="user-1",
        question="복잡한 에이전틱 질문",
        session_id=initial_session.get_session_id(),
    )
    await consumer._process(message)

    session_repo_mock.update.assert_awaited_once()
    updated_session = session_repo_mock.update.await_args[0][0]
    assert len(updated_session.turns) == 2
    assert updated_session.turns[1].role == "assistant"
    assert updated_session.turns[1].content == "에이전틱 답변입니다."
    assert updated_session.turns[1].confidence == 0.85
    assert updated_session.turns[1].missing == ["추가 설명"]
    assert updated_session.turns[1].iterations == 1


@pytest.mark.asyncio
async def test_ask_requested_consumer_simple_persists_confidence_and_publishes_done():
    redis_mock = MagicMock(spec=Redis)
    redis_mock.xadd = AsyncMock()
    redis_mock.get = AsyncMock(return_value=None)

    sources_data = [
        {
            "fileName": "test.pdf",
            "chunkIndex": 0,
            "documentId": "doc-1",
            "snippet": "테스트 요약",
            "score": 0.88,
        }
    ]
    confidence_data = {"confidence": 0.88, "missing": []}

    ask_use_case_mock = MagicMock(spec=AskUseCase)

    async def fake_ask_execute(_command) -> AsyncIterator[str]:
        yield f"__SOURCES:{json.dumps(sources_data)}"
        yield f"__CONFIDENCE:{json.dumps(confidence_data)}"
        yield "단순 답변입니다."

    ask_use_case_mock.execute.side_effect = fake_ask_execute

    router_mock = MagicMock(spec=QueryComplexityRouter)
    router_mock.route.return_value = "simple"

    validator_mock = MagicMock(spec=RagContentValidator)
    validator_mock.inspect_input.return_value = GuardrailVerdict.allow()

    initial_session = ConversationSession.create("user-1", "질문입니다")
    session_repo_mock = MagicMock(spec=ConversationSessionRepository)
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
        job_id="test-job-simple-confidence",
        user_id="user-1",
        question="질문입니다",
        session_id=initial_session.get_session_id(),
    )
    await consumer._process(message)

    # 1. session_repo.update 검증
    session_repo_mock.update.assert_awaited_once()
    updated_session = session_repo_mock.update.await_args[0][0]
    assert len(updated_session.turns) == 2
    assert updated_session.turns[0].iterations is None
    assert updated_session.turns[1].role == "assistant"
    assert updated_session.turns[1].content == "단순 답변입니다."
    assert updated_session.turns[1].sources == sources_data
    assert updated_session.turns[1].confidence == 0.88
    assert updated_session.turns[1].missing == []
    assert updated_session.turns[1].iterations is None

    # 2. done 이벤트에 confidence 메타데이터 포함 검증
    done_calls = [
        call
        for call in redis_mock.xadd.await_args_list
        if call.args[1].get("type") == "done"
    ]
    assert len(done_calls) == 1
    assert "data" in done_calls[0].args[1]
    done_data = json.loads(done_calls[0].args[1]["data"])
    assert done_data == {"confidence": 0.88, "missing": []}


@pytest.mark.asyncio
async def test_ask_requested_consumer_simple_persists_zero_confidence_when_no_chunks():
    redis_mock = MagicMock(spec=Redis)
    redis_mock.xadd = AsyncMock()
    redis_mock.get = AsyncMock(return_value=None)

    confidence_data = {"confidence": 0.0, "missing": ["관련 지식베이스 문서를 찾지 못함"]}

    ask_use_case_mock = MagicMock(spec=AskUseCase)

    async def fake_ask_execute(_command) -> AsyncIterator[str]:
        yield f"__CONFIDENCE:{json.dumps(confidence_data, ensure_ascii=False)}"
        yield "근거 없는 답변입니다."

    ask_use_case_mock.execute.side_effect = fake_ask_execute

    router_mock = MagicMock(spec=QueryComplexityRouter)
    router_mock.route.return_value = "simple"

    validator_mock = MagicMock(spec=RagContentValidator)
    validator_mock.inspect_input.return_value = GuardrailVerdict.allow()

    initial_session = ConversationSession.create("user-1", "문서 없는 질문")
    session_repo_mock = MagicMock(spec=ConversationSessionRepository)
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
        job_id="test-job-simple-zero-confidence",
        user_id="user-1",
        question="문서 없는 질문",
        session_id=initial_session.get_session_id(),
    )
    await consumer._process(message)

    # 1. session_repo.update 검증
    session_repo_mock.update.assert_awaited_once()
    updated_session = session_repo_mock.update.await_args[0][0]
    assert len(updated_session.turns) == 2
    assert updated_session.turns[1].role == "assistant"
    assert updated_session.turns[1].content == "근거 없는 답변입니다."
    assert updated_session.turns[1].sources is None
    assert updated_session.turns[1].confidence == 0.0
    assert updated_session.turns[1].missing == ["관련 지식베이스 문서를 찾지 못함"]

    # 2. done 이벤트에 0.0 confidence 메타데이터 포함 검증
    done_calls = [
        call
        for call in redis_mock.xadd.await_args_list
        if call.args[1].get("type") == "done"
    ]
    assert len(done_calls) == 1
    assert "data" in done_calls[0].args[1]
    done_data = json.loads(done_calls[0].args[1]["data"])
    assert done_data == {"confidence": 0.0, "missing": ["관련 지식베이스 문서를 찾지 못함"]}


async def test_ask_requested_consumer_agentic_cancelled_stops_streaming():
    redis_mock = MagicMock(spec=Redis)
    redis_mock.xadd = AsyncMock()

    # 첫 토큰은 정상 수신, 두 번째 토큰 수신 시점에서 취소 감지
    redis_mock.get = AsyncMock(side_effect=[None, b"1", b"1"])

    agentic_use_case_mock = MagicMock(spec=AgenticAskUseCase)

    async def fake_agentic_stream(_command) -> AsyncIterator[str]:
        yield "첫번째 토큰 "
        yield "두번째 토큰 "
        yield "세번째 토큰 "

    agentic_use_case_mock.execute.side_effect = fake_agentic_stream

    router_mock = MagicMock(spec=QueryComplexityRouter)
    router_mock.route.return_value = "complex"

    validator_mock = MagicMock(spec=RagContentValidator)
    validator_mock.inspect_input.return_value = GuardrailVerdict.allow()

    initial_session = ConversationSession.create("user-1", "복잡한 질문")
    session_repo_mock = MagicMock(spec=ConversationSessionRepository)
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
        job_id="test-job-agentic-cancel",
        user_id="user-1",
        question="복잡한 질문",
        session_id=initial_session.get_session_id(),
    )
    await consumer._process(message)

    # 취소로 인해 1개 토큰만 publish_token 되고 이후 토큰은 중단됨
    token_calls = [
        call
        for call in redis_mock.xadd.await_args_list
        if call.args[1].get("type") == "token"
    ]
    assert len(token_calls) == 1
    assert token_calls[0].args[1]["data"] == "첫번째 토큰 "


async def test_ask_requested_consumer_complex_persists_latest_multi_turn_sources():
    redis_mock = MagicMock(spec=Redis)
    redis_mock.xadd = AsyncMock()
    redis_mock.get = AsyncMock(return_value=None)

    agentic_use_case_mock = MagicMock(spec=AgenticAskUseCase)

    async def fake_agentic_execute(command) -> AsyncIterator[str]:
        # 1회차 출처 방출
        sources_iter1 = [
            {
                "documentId": "doc-A",
                "chunkIndex": 0,
                "fileName": "a.pdf",
                "snippet": "A",
                "score": 0.7,
            }
        ]
        yield f"__SOURCES:{json.dumps(sources_iter1)}"

        # 2회차 출처 방출 (누적 및 갱신)
        sources_iter2 = [
            {
                "documentId": "doc-A",
                "chunkIndex": 0,
                "fileName": "a.pdf",
                "snippet": "A",
                "score": 0.9,
            },
            {
                "documentId": "doc-B",
                "chunkIndex": 1,
                "fileName": "b.pdf",
                "snippet": "B",
                "score": 0.8,
            },
        ]
        yield f"__SOURCES:{json.dumps(sources_iter2)}"

        yield "최종 다회차 답변입니다."

    agentic_use_case_mock.execute.side_effect = fake_agentic_execute

    router_mock = MagicMock(spec=QueryComplexityRouter)
    router_mock.route.return_value = "complex"

    validator_mock = MagicMock(spec=RagContentValidator)
    validator_mock.inspect_input.return_value = GuardrailVerdict.allow()

    initial_session = ConversationSession.create("user-1", "복잡한 에이전틱 질문")
    session_repo_mock = MagicMock(spec=ConversationSessionRepository)
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
        job_id="test-job-multi-sources",
        user_id="user-1",
        question="복잡한 에이전틱 질문",
        session_id=initial_session.get_session_id(),
    )
    await consumer._process(message)

    # sources redis 이벤트 발행 확인 (2회 발행되었는지)
    sources_calls = [
        call
        for call in redis_mock.xadd.await_args_list
        if call.args[1].get("type") == "sources"
    ]
    assert len(sources_calls) == 2

    # 세션에 최종 저장된 sources는 2회차 누적 출처 목록이어야 함
    session_repo_mock.update.assert_awaited_once()
    updated_session = session_repo_mock.update.await_args[0][0]
    assert len(updated_session.turns) == 2
    assistant_turn = updated_session.turns[1]
    assert assistant_turn.sources is not None
    assert len(assistant_turn.sources) == 2
    assert assistant_turn.sources[0]["documentId"] == "doc-A"
    assert assistant_turn.sources[0]["score"] == 0.9
    assert assistant_turn.sources[1]["documentId"] == "doc-B"


@pytest.mark.asyncio
async def test_ask_requested_consumer_complex_passes_session_id_to_agentic_ask_command():
    redis_mock = MagicMock(spec=Redis)
    redis_mock.xadd = AsyncMock()
    redis_mock.get = AsyncMock(return_value=None)

    captured_commands = []
    agentic_use_case_mock = MagicMock(spec=AgenticAskUseCase)

    async def fake_agentic_execute(command) -> AsyncIterator[str]:
        captured_commands.append(command)
        yield "후속 복합 질문 답변입니다."

    agentic_use_case_mock.execute.side_effect = fake_agentic_execute

    router_mock = MagicMock(spec=QueryComplexityRouter)
    router_mock.route.return_value = "complex"

    validator_mock = MagicMock(spec=RagContentValidator)
    validator_mock.inspect_input.return_value = GuardrailVerdict.allow()

    existing_session = ConversationSession.create("user-1", "이전 질문").append_turn(
        "이전 질문", "이전 답변"
    )
    session_repo_mock = MagicMock(spec=ConversationSessionRepository)
    session_repo_mock.find_by_id = AsyncMock(return_value=existing_session)
    session_repo_mock.persist = AsyncMock(return_value=existing_session)
    session_repo_mock.update = AsyncMock(return_value=existing_session)

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
        job_id="test-job-complex-session-id",
        user_id="user-1",
        question="거기서 환불 조건과 해지 절차를 비교해 줘",
        session_id=existing_session.get_session_id(),
        conversation_history=None,
    )
    await consumer._process(message)

    assert len(captured_commands) == 1
    assert captured_commands[0].session_id == existing_session.get_session_id()


@pytest.mark.asyncio
async def test_ask_requested_consumer_complex_passes_none_session_id_when_session_not_found():
    redis_mock = MagicMock(spec=Redis)
    redis_mock.xadd = AsyncMock()
    redis_mock.get = AsyncMock(return_value=None)

    captured_commands = []
    agentic_use_case_mock = MagicMock(spec=AgenticAskUseCase)

    async def fake_agentic_execute(command) -> AsyncIterator[str]:
        captured_commands.append(command)
        yield "세션 미존재 복합 질문 답변입니다."

    agentic_use_case_mock.execute.side_effect = fake_agentic_execute

    router_mock = MagicMock(spec=QueryComplexityRouter)
    router_mock.route.return_value = "complex"

    validator_mock = MagicMock(spec=RagContentValidator)
    validator_mock.inspect_input.return_value = GuardrailVerdict.allow()

    session_repo_mock = MagicMock(spec=ConversationSessionRepository)
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
        job_id="test-job-complex-missing-session",
        user_id="user-1",
        question="거기서 환불 조건과 해지 절차를 비교해 줘",
        session_id="missing-session-id",
        conversation_history=None,
    )
    await consumer._process(message)

    session_repo_mock.find_by_id.assert_awaited_once_with("missing-session-id")
    assert len(captured_commands) == 1
    assert captured_commands[0].session_id is None


@pytest.mark.asyncio
async def test_ask_requested_consumer_complex_exhausted_budget_publishes_fallback():
    # 수용 기준 2 엔드투엔드(Consumer + AgenticAskUseCase) 검증:
    # 1회차 실행 중 지식 문서 청크 2건(0.85, 0.72) 검색 후 예산 소진으로 조기 종료 시
    # confidence=0.85, missing=["답변 검증 예산 소진으로 상세 비평 생략"]이
    # progress 이벤트, done 이벤트, 세션 저장에 온전하게 반영되는지 확인
    from ai_service.knowledge.schemas import (
        SimilaritySearchResult,
        VectorDocumentMetadata,
    )
    from ai_service.prompt.schemas import PromptTemplate
    from ai_service.prompt.service import PromptService
    from ai_service.rag.application.critique_generator_service import (
        CritiqueGeneratorService,
    )
    from ai_service.rag.application.hybrid_search_use_case import (
        HybridSearchResult,
        HybridSearchUseCase,
    )
    from ai_service.rag.application.query_refiner_service import QueryRefinerService

    redis_mock = MagicMock(spec=Redis)
    redis_mock.xadd = AsyncMock()
    redis_mock.get = AsyncMock(return_value=None)

    chunks = [
        SimilaritySearchResult(
            text="문서 내용 1",
            score=0.72,
            metadata=VectorDocumentMetadata(document_id="doc-1", file_name="f1.pdf", chunk_index=0),
        ),
        SimilaritySearchResult(
            text="문서 내용 2",
            score=0.85,
            metadata=VectorDocumentMetadata(document_id="doc-2", file_name="f2.pdf", chunk_index=1),
        ),
    ]

    mock_hybrid_search = AsyncMock(spec=HybridSearchUseCase)
    mock_hybrid_search.execute.return_value = HybridSearchResult(
        chunks=chunks, query_embedding=[0.1, 0.2]
    )

    mock_llm_gateway = MagicMock()

    async def fake_stream(_command) -> AsyncIterator[str]:
        yield "답변 내용입니다.\n"

    mock_llm_gateway.stream.side_effect = fake_stream

    mock_prompt_service = AsyncMock(spec=PromptService)
    mock_prompt_service.execute.return_value = PromptTemplate.create(
        "rag-qa-system",
        "Context: {{context}}",
        ["context"],
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

    session = ConversationSession.create("user-1", "복합 질문")
    session_repo_mock = MagicMock(spec=ConversationSessionRepository)
    session_repo_mock.find_by_id = AsyncMock(return_value=session)
    session_repo_mock.persist = AsyncMock(return_value=session)
    session_repo_mock.update = AsyncMock(return_value=None)

    composition = RagComposition(
        ask_use_case=MagicMock(spec=AskUseCase),
        agentic_ask_use_case=agentic_use_case,
        query_complexity_router=router_mock,
        rag_validator=RagContentValidator(),
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
        job_id="test-job-fallback-confidence",
        user_id="user-1",
        question="1회차 예산 소진 복합 질문",
        session_id=session.get_session_id(),
    )

    await consumer._process(message)

    # 1. progress 이벤트 확인
    progress_calls = [
        call for call in redis_mock.xadd.await_args_list if call.args[1].get("type") == "progress"
    ]
    assert len(progress_calls) >= 2
    last_progress = json.loads(progress_calls[-1].args[1]["data"])
    assert last_progress["confidence"] == 0.85
    assert last_progress["missing"] == ["답변 검증 예산 소진으로 상세 비평 생략"]

    # 2. done 이벤트 확인
    done_calls = [
        call for call in redis_mock.xadd.await_args_list if call.args[1].get("type") == "done"
    ]
    assert len(done_calls) == 1
    done_data = json.loads(done_calls[0].args[1]["data"])
    assert done_data["confidence"] == 0.85
    assert done_data["missing"] == ["답변 검증 예산 소진으로 상세 비평 생략"]
    assert done_data["iterations"] == 1

    # 3. session.append_turn 저장 확인
    session_repo_mock.update.assert_awaited_once()
    updated_session: ConversationSession = session_repo_mock.update.await_args.args[0]
    last_turn = updated_session.turns[-1]
    assert last_turn.confidence == 0.85
    assert last_turn.missing == ["답변 검증 예산 소진으로 상세 비평 생략"]
    assert last_turn.iterations == 1


@pytest.mark.asyncio
async def test_ask_requested_consumer_complex_exhausted_budget_no_chunks_zero_confidence():
    # 수용 기준 3 엔드투엔드 검증: 청크가 0건이고 예산 소진 시 confidence=0.0, 사유 반영 확인
    from ai_service.prompt.schemas import PromptTemplate
    from ai_service.prompt.service import PromptService
    from ai_service.rag.application.critique_generator_service import (
        CritiqueGeneratorService,
    )
    from ai_service.rag.application.hybrid_search_use_case import (
        HybridSearchResult,
        HybridSearchUseCase,
    )
    from ai_service.rag.application.query_refiner_service import QueryRefinerService

    redis_mock = MagicMock(spec=Redis)
    redis_mock.xadd = AsyncMock()
    redis_mock.get = AsyncMock(return_value=None)

    mock_hybrid_search = AsyncMock(spec=HybridSearchUseCase)
    mock_hybrid_search.execute.return_value = HybridSearchResult(
        chunks=[], query_embedding=[0.1, 0.2]
    )

    mock_llm_gateway = MagicMock()

    async def fake_stream(_command) -> AsyncIterator[str]:
        yield "문서 없이 생성된 답변.\n"

    mock_llm_gateway.stream.side_effect = fake_stream

    mock_prompt_service = AsyncMock(spec=PromptService)
    mock_prompt_service.execute.return_value = PromptTemplate.create(
        "rag-qa-system",
        "Context: {{context}}",
        ["context"],
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

    session = ConversationSession.create("user-1", "문서 없는 복합 질문")
    session_repo_mock = MagicMock(spec=ConversationSessionRepository)
    session_repo_mock.find_by_id = AsyncMock(return_value=session)
    session_repo_mock.persist = AsyncMock(return_value=session)
    session_repo_mock.update = AsyncMock(return_value=None)

    composition = RagComposition(
        ask_use_case=MagicMock(spec=AskUseCase),
        agentic_ask_use_case=agentic_use_case,
        query_complexity_router=router_mock,
        rag_validator=RagContentValidator(),
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
        job_id="test-job-no-chunks-exhausted",
        user_id="user-1",
        question="문서 없는 1회차 예산 소진 복합 질문",
        session_id=session.get_session_id(),
    )

    await consumer._process(message)

    done_calls = [
        call for call in redis_mock.xadd.await_args_list if call.args[1].get("type") == "done"
    ]
    assert len(done_calls) == 1
    done_data = json.loads(done_calls[0].args[1]["data"])
    assert done_data["confidence"] == 0.0
    assert done_data["missing"] == ["관련 지식베이스 문서를 찾지 못함"]
    assert done_data["iterations"] == 1



