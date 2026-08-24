from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from ai_service.main import app
from ai_service.rag.domain.model.conversation_session import (
    ConversationSession,
    RestoreProps,
    TurnRecord,
)
from ai_service.rag.domain.repository.conversation_session_repository import (
    IConversationSessionRepository,
)
from ai_service.rag.presentation.deps import get_conversation_session_repository


@pytest.fixture(autouse=True)
def _clear_overrides():
    yield
    app.dependency_overrides.clear()


async def test_get_session_returns_turns_with_metadata() -> None:
    now = datetime.now(UTC)
    sources: list[dict[str, Any]] = [
        {
            "fileName": "refund.pdf",
            "chunkIndex": 0,
            "documentId": "doc-1",
            "snippet": "환불 규정 본문",
        },
        {
            "fileName": "terms.pdf",
            "chunkIndex": 1,
            "documentId": "doc-2",
            "snippet": "이용약관",
        },
    ]
    props = RestoreProps(
        session_id="session-meta-123",
        user_id="user-1",
        title="메타데이터 세션",
        turns=[
            TurnRecord(role="user", content="환불 규정 질문", created_at=now),
            TurnRecord(
                role="assistant",
                content="환불 가능합니다.",
                created_at=now,
                sources=sources,
                confidence=0.92,
                missing=["예외 규정"],
            ),
        ],
        created_at=now,
        updated_at=now,
    )
    session = ConversationSession.restore(props)

    repo_mock = MagicMock(spec=IConversationSessionRepository)
    repo_mock.find_by_id = AsyncMock(return_value=session)
    app.dependency_overrides[get_conversation_session_repository] = lambda: repo_mock

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/rag/sessions/session-meta-123")

    assert response.status_code == 200
    body = response.json()
    assert body["sessionId"] == "session-meta-123"
    assert len(body["turns"]) == 2
    assert body["turns"][0]["role"] == "user"
    assert body["turns"][0]["sources"] is None
    assert body["turns"][0]["confidence"] is None
    assert body["turns"][0]["missing"] is None

    assert body["turns"][1]["role"] == "assistant"
    assert body["turns"][1]["content"] == "환불 가능합니다."
    assert body["turns"][1]["sources"] == sources
    assert body["turns"][1]["confidence"] == 0.92
    assert body["turns"][1]["missing"] == ["예외 규정"]


async def test_get_session_legacy_data_returns_none_metadata() -> None:
    now = datetime.now(UTC)
    props = RestoreProps(
        session_id="session-legacy-123",
        user_id="user-1",
        title="레거시 세션",
        turns=[
            TurnRecord(role="user", content="질문", created_at=now),
            TurnRecord(role="assistant", content="답변", created_at=now),
        ],
        created_at=now,
        updated_at=now,
    )
    session = ConversationSession.restore(props)

    repo_mock = MagicMock(spec=IConversationSessionRepository)
    repo_mock.find_by_id = AsyncMock(return_value=session)
    app.dependency_overrides[get_conversation_session_repository] = lambda: repo_mock

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/rag/sessions/session-legacy-123")

    assert response.status_code == 200
    body = response.json()
    assert body["sessionId"] == "session-legacy-123"
    assert len(body["turns"]) == 2
    assert body["turns"][1]["sources"] is None
    assert body["turns"][1]["confidence"] is None
    assert body["turns"][1]["missing"] is None


async def test_get_session_not_found_returns_null() -> None:
    repo_mock = MagicMock(spec=IConversationSessionRepository)
    repo_mock.find_by_id = AsyncMock(return_value=None)
    app.dependency_overrides[get_conversation_session_repository] = lambda: repo_mock

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/rag/sessions/non-existent")

    assert response.status_code == 200
    assert response.json() is None
