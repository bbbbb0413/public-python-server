from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import MagicMock

import pytest

from ai_service.rag.application.ask_command import AskCommand
from ai_service.rag.domain.model.conversation_session import ConversationSession
from ai_service.rag.infrastructure.messaging.ask_requested_consumer import (
    _CANCEL_CHECK_INTERVAL,
    AskRequestedConsumer,
    AskRequestedMessage,
)


class FakeRedis:
    def __init__(self) -> None:
        self.data: dict[str, dict[str, Any]] = {}
        self.keys: set[str] = set()
        self.stream_events: list[dict[str, Any]] = []

    async def hget(self, key: str, field: str) -> Any:
        return self.data.get(key, {}).get(field)

    async def hset(
        self,
        key: str,
        field: str | None = None,
        value: Any = None,
        mapping: dict[str, Any] | None = None,
    ) -> None:
        if key not in self.data:
            self.data[key] = {}
        if mapping:
            self.data[key].update(mapping)
        if field is not None and value is not None:
            self.data[key][field] = value

    async def exists(self, *keys: str) -> int:
        return sum(1 for k in keys if k in self.keys or k in self.data)

    async def xadd(self, stream: str, fields: dict[str, Any]) -> str:
        self.stream_events.append({"stream": stream, **fields})
        return "1-0"


class FakeAskUseCase:
    def __init__(self, tokens: list[str]) -> None:
        self._tokens = tokens

    async def execute(self, command: AskCommand) -> AsyncIterator[str]:
        for token in self._tokens:
            yield token


class FakeSessionRepo:
    def __init__(self) -> None:
        self.sessions: dict[str, ConversationSession] = {}
        self.updated_sessions: list[ConversationSession] = []

    async def find_by_id(self, session_id: str) -> ConversationSession | None:
        return self.sessions.get(session_id)

    async def persist(self, session: ConversationSession) -> ConversationSession:
        self.sessions[session.get_session_id()] = session
        return session

    async def update(self, session: ConversationSession) -> None:
        self.updated_sessions.append(session)


def create_fake_composition(tokens: list[str]) -> tuple[MagicMock, FakeSessionRepo]:
    composition = MagicMock()
    composition.guardrail_enabled = False
    composition.query_complexity_router.route.return_value = "simple"
    composition.hyde_max_query_words = 10
    composition.ask_use_case = FakeAskUseCase(tokens)

    session_repo = FakeSessionRepo()
    composition.session_repo = session_repo

    scanner = MagicMock()
    scanner.mask.side_effect = lambda text: text
    composition.secret_pii_scanner = scanner

    return composition, session_repo


@pytest.mark.asyncio
async def test_ask_requested_consumer_completes_normally_when_not_cancelled() -> None:
    tokens = [f"token_{i}" for i in range(25)]
    composition, session_repo = create_fake_composition(tokens)
    fake_redis = FakeRedis()

    consumer = AskRequestedConsumer(
        brokers="localhost:9092",
        redis_client=fake_redis,  # type: ignore[arg-type]
        composition=composition,
    )

    message = AskRequestedMessage(
        job_id="job-1",
        user_id="user-1",
        question="테스트 질문",
    )

    await consumer._process(message)

    token_events = [e for e in fake_redis.stream_events if e.get("type") == "token"]
    done_events = [e for e in fake_redis.stream_events if e.get("type") == "done"]

    assert len(token_events) == 25
    assert len(done_events) == 1
    assert len(session_repo.updated_sessions) == 1


@pytest.mark.asyncio
async def test_ask_requested_consumer_stops_when_cancelled_at_interval() -> None:
    tokens = [f"token_{i}" for i in range(25)]
    composition, session_repo = create_fake_composition(tokens)
    fake_redis = FakeRedis()

    await fake_redis.hset("job:job-cancel", "status", "cancelled")

    consumer = AskRequestedConsumer(
        brokers="localhost:9092",
        redis_client=fake_redis,  # type: ignore[arg-type]
        composition=composition,
    )

    message = AskRequestedMessage(
        job_id="job-cancel",
        user_id="user-1",
        question="취소 테스트 질문",
    )

    await consumer._process(message)

    token_events = [e for e in fake_redis.stream_events if e.get("type") == "token"]
    done_events = [e for e in fake_redis.stream_events if e.get("type") == "done"]

    assert len(token_events) < 25
    assert len(token_events) == _CANCEL_CHECK_INTERVAL - 1
    assert len(done_events) == 1
    assert len(session_repo.updated_sessions) == 0


@pytest.mark.asyncio
async def test_is_cancelled_detects_status_and_keys() -> None:
    fake_redis = FakeRedis()
    composition, _ = create_fake_composition([])
    consumer = AskRequestedConsumer(
        brokers="localhost:9092",
        redis_client=fake_redis,  # type: ignore[arg-type]
        composition=composition,
    )

    assert await consumer._is_cancelled("job-clean") is False

    await fake_redis.hset("job:job-status-str", "status", "cancelled")
    assert await consumer._is_cancelled("job-status-str") is True

    await fake_redis.hset("job:job-status-bytes", "status", b"cancelled")
    assert await consumer._is_cancelled("job-status-bytes") is True

    await fake_redis.hset("job:job-cancelled-flag", "cancelled", "true")
    assert await consumer._is_cancelled("job-cancelled-flag") is True

    fake_redis.keys.add("job:job-cancel-key:cancelled")
    assert await consumer._is_cancelled("job-cancel-key") is True


@pytest.mark.asyncio
async def test_is_cancelled_handles_redis_exception_gracefully() -> None:
    fake_redis = FakeRedis()
    composition, _ = create_fake_composition([])
    consumer = AskRequestedConsumer(
        brokers="localhost:9092",
        redis_client=fake_redis,  # type: ignore[arg-type]
        composition=composition,
    )

    async def raise_error(*args: Any, **kwargs: Any) -> Any:
        raise ConnectionError("Redis connection lost")

    fake_redis.hget = raise_error  # type: ignore[assignment]
    assert await consumer._is_cancelled("job-err") is False


@pytest.mark.asyncio
async def test_ask_requested_consumer_agentic_flow_stops_when_cancelled() -> None:
    composition = MagicMock()
    composition.guardrail_enabled = False
    composition.query_complexity_router.route.return_value = "complex"
    composition.hyde_max_query_words = 10
    tokens = [f"agentic_token_{i}" for i in range(25)]
    composition.agentic_ask_use_case = FakeAskUseCase(tokens)

    session_repo = FakeSessionRepo()
    composition.session_repo = session_repo

    scanner = MagicMock()
    scanner.mask.side_effect = lambda text: text
    composition.secret_pii_scanner = scanner

    fake_redis = FakeRedis()
    await fake_redis.hset("job:agentic-cancel", "status", "cancelled")

    consumer = AskRequestedConsumer(
        brokers="localhost:9092",
        redis_client=fake_redis,  # type: ignore[arg-type]
        composition=composition,
    )

    message = AskRequestedMessage(
        job_id="agentic-cancel",
        user_id="user-1",
        question="복잡한 에이전틱 질문",
    )

    await consumer._process(message)

    token_events = [e for e in fake_redis.stream_events if e.get("type") == "token"]
    done_events = [e for e in fake_redis.stream_events if e.get("type") == "done"]

    assert len(token_events) == _CANCEL_CHECK_INTERVAL - 1
    assert len(done_events) == 1
    assert len(session_repo.updated_sessions) == 0

