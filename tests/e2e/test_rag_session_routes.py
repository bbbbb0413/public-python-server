from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from ai_service.main import app
from ai_service.rag.dependencies import get_conversation_session_repository
from ai_service.rag.repository import ConversationSessionRepository
from ai_service.rag.schemas import (
    ConversationSession,
    RestoreProps,
    TurnRecord,
)


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

    repo_mock = MagicMock(spec=ConversationSessionRepository)
    repo_mock.find_by_id_for_user = AsyncMock(return_value=session)
    app.dependency_overrides[get_conversation_session_repository] = lambda: repo_mock

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/rag/sessions/session-meta-123?userId=user-1")

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

    repo_mock = MagicMock(spec=ConversationSessionRepository)
    repo_mock.find_by_id_for_user = AsyncMock(return_value=session)
    app.dependency_overrides[get_conversation_session_repository] = lambda: repo_mock

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/rag/sessions/session-legacy-123?userId=user-1")

    assert response.status_code == 200
    body = response.json()
    assert body["sessionId"] == "session-legacy-123"
    assert len(body["turns"]) == 2
    assert body["turns"][1]["sources"] is None
    assert body["turns"][1]["confidence"] is None
    assert body["turns"][1]["missing"] is None


async def test_get_session_not_found_returns_404() -> None:
    repo_mock = MagicMock(spec=ConversationSessionRepository)
    repo_mock.find_by_id_for_user = AsyncMock(return_value=None)
    app.dependency_overrides[get_conversation_session_repository] = lambda: repo_mock

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/rag/sessions/non-existent?userId=user-1")

    assert response.status_code == 404


async def test_get_session_requires_user_id() -> None:
    """userId 없이 세션을 열 수 없다. 게이트웨이가 반드시 채워 보내야 한다."""
    repo_mock = MagicMock(spec=ConversationSessionRepository)
    app.dependency_overrides[get_conversation_session_repository] = lambda: repo_mock

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/rag/sessions/session-meta-123")

    assert response.status_code == 422


async def test_get_session_of_another_user_is_indistinguishable_from_missing() -> None:
    """남의 세션과 없는 세션이 같은 404 여야 세션 존재 여부가 새지 않는다."""
    repo_mock = MagicMock(spec=ConversationSessionRepository)
    # 소유권 조건이 쿼리에 있으므로 남의 세션은 애초에 None 으로 돌아온다.
    repo_mock.find_by_id_for_user = AsyncMock(return_value=None)
    app.dependency_overrides[get_conversation_session_repository] = lambda: repo_mock

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        other = await client.get("/rag/sessions/session-meta-123?userId=침입자")
        missing = await client.get("/rag/sessions/없는-세션?userId=침입자")

    assert other.status_code == missing.status_code == 404
    assert other.json() == missing.json()
    repo_mock.find_by_id_for_user.assert_awaited_with("없는-세션", "침입자")


async def test_delete_session_scopes_to_owner() -> None:
    repo_mock = MagicMock(spec=ConversationSessionRepository)
    repo_mock.delete_by_id_for_user = AsyncMock(return_value=True)
    app.dependency_overrides[get_conversation_session_repository] = lambda: repo_mock

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.delete("/rag/sessions/session-1?userId=user-1")

    assert response.status_code == 204
    repo_mock.delete_by_id_for_user.assert_awaited_once_with("session-1", "user-1")


async def test_delete_session_of_another_user_returns_404() -> None:
    """지운 것이 없으면 204 로 성공한 척하지 않는다."""
    repo_mock = MagicMock(spec=ConversationSessionRepository)
    repo_mock.delete_by_id_for_user = AsyncMock(return_value=False)
    app.dependency_overrides[get_conversation_session_repository] = lambda: repo_mock

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.delete("/rag/sessions/남의-세션?userId=침입자")

    assert response.status_code == 404
