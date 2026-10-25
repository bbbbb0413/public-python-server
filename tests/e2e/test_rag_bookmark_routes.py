from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from ai_service.main import app
from ai_service.rag.dependencies import (
    get_bookmark_repository,
    get_conversation_session_repository,
)
from ai_service.rag.repository import BookmarkRepository, ConversationSessionRepository
from ai_service.rag.schemas import (
    ConversationSession,
    RestoreProps,
    TurnRecord,
)


@pytest.fixture(autouse=True)
def _clear_overrides():
    session_repo_mock = MagicMock(spec=ConversationSessionRepository)
    session_repo_mock.find_by_id_for_user = AsyncMock(return_value=None)
    app.dependency_overrides[get_conversation_session_repository] = lambda: session_repo_mock

    bookmark_repo_mock = MagicMock(spec=BookmarkRepository)
    app.dependency_overrides[get_bookmark_repository] = lambda: bookmark_repo_mock
    yield
    app.dependency_overrides.clear()


def _create_sample_session(
    session_id: str = "sess-1", user_id: str = "user-1"
) -> ConversationSession:
    now = datetime.now(UTC)
    props = RestoreProps(
        session_id=session_id,
        user_id=user_id,
        title="파이프라인 질의",
        turns=[
            TurnRecord(role="user", content="파이프라인이 뭔가요?", created_at=now),
            TurnRecord(role="assistant", content="RAG 검색 결과입니다.", created_at=now),
        ],
        created_at=now,
        updated_at=now,
    )
    return ConversationSession.restore(props)


async def test_create_bookmark_turn_success() -> None:
    session_repo_mock = MagicMock(spec=ConversationSessionRepository)
    session_repo_mock.find_by_id_for_user = AsyncMock(return_value=_create_sample_session())
    app.dependency_overrides[get_conversation_session_repository] = lambda: session_repo_mock

    bookmark_repo_mock = MagicMock(spec=BookmarkRepository)
    bookmark_repo_mock.create = AsyncMock(return_value="bm-test-1")
    app.dependency_overrides[get_bookmark_repository] = lambda: bookmark_repo_mock

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/rag/bookmarks?userId=user-1",
            json={"sessionId": "sess-1", "turnIndex": 1, "note": "보관 메모"},
        )

    assert response.status_code in (200, 201)
    body = response.json()
    assert body["id"] == "bm-test-1"
    assert body["sessionId"] == "sess-1"
    assert body["turnIndex"] == 1
    assert body["content"] == "RAG 검색 결과입니다."
    assert body["title"] == "파이프라인 질의"
    assert body["note"] == "보관 메모"
    assert "createdAt" in body


async def test_create_bookmark_session_level_when_turn_index_missing() -> None:
    session_repo_mock = MagicMock(spec=ConversationSessionRepository)
    session_repo_mock.find_by_id_for_user = AsyncMock(return_value=_create_sample_session())
    app.dependency_overrides[get_conversation_session_repository] = lambda: session_repo_mock

    bookmark_repo_mock = MagicMock(spec=BookmarkRepository)
    bookmark_repo_mock.create = AsyncMock(return_value="bm-test-2")
    app.dependency_overrides[get_bookmark_repository] = lambda: bookmark_repo_mock

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/rag/bookmarks?userId=user-1",
            json={"sessionId": "sess-1"},
        )

    assert response.status_code in (200, 201)
    body = response.json()
    assert body["id"] == "bm-test-2"
    assert body["turnIndex"] is None
    assert body["content"] is None
    assert body["title"] == "파이프라인 질의"


async def test_create_bookmark_nonexistent_session_graceful() -> None:
    session_repo_mock = MagicMock(spec=ConversationSessionRepository)
    session_repo_mock.find_by_id_for_user = AsyncMock(return_value=None)
    app.dependency_overrides[get_conversation_session_repository] = lambda: session_repo_mock

    bookmark_repo_mock = MagicMock(spec=BookmarkRepository)
    bookmark_repo_mock.create = AsyncMock(return_value="bm-test-3")
    app.dependency_overrides[get_bookmark_repository] = lambda: bookmark_repo_mock

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/rag/bookmarks?userId=user-1",
            json={"sessionId": "sess-nonexistent", "turnIndex": 0},
        )

    assert response.status_code in (200, 201)
    body = response.json()
    assert body["id"] == "bm-test-3"
    assert body["sessionId"] == "sess-nonexistent"


async def test_get_bookmarks_with_keyword_query() -> None:
    now = datetime.now(UTC)
    bookmark_repo_mock = MagicMock(spec=BookmarkRepository)
    bookmark_repo_mock.find_by_user_id = AsyncMock(
        return_value=[
            {
                "id": "bm-1",
                "sessionId": "sess-1",
                "turnIndex": 1,
                "title": "지식베이스 질의",
                "content": "파이프라인 안내",
                "note": "메모",
                "createdAt": now,
            }
        ]
    )
    app.dependency_overrides[get_bookmark_repository] = lambda: bookmark_repo_mock

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/rag/bookmarks?userId=user-1&q=파이프라인")

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["id"] == "bm-1"
    assert body[0]["title"] == "지식베이스 질의"
    assert body[0]["content"] == "파이프라인 안내"
    bookmark_repo_mock.find_by_user_id.assert_awaited_once_with("user-1", 1, 20, "파이프라인")


async def test_delete_bookmark_owner_success() -> None:
    bookmark_repo_mock = MagicMock(spec=BookmarkRepository)
    bookmark_repo_mock.delete_by_id_for_user = AsyncMock(return_value=True)
    app.dependency_overrides[get_bookmark_repository] = lambda: bookmark_repo_mock

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.delete("/rag/bookmarks/bm-test-1?userId=user-1")

    assert response.status_code == 204
    bookmark_repo_mock.delete_by_id_for_user.assert_awaited_once_with("bm-test-1", "user-1")


async def test_delete_bookmark_idor_defense_returns_404() -> None:
    bookmark_repo_mock = MagicMock(spec=BookmarkRepository)
    bookmark_repo_mock.delete_by_id_for_user = AsyncMock(return_value=False)
    app.dependency_overrides[get_bookmark_repository] = lambda: bookmark_repo_mock

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.delete("/rag/bookmarks/bm-user2-99?userId=user-1")

    assert response.status_code == 404
    body = response.json()
    assert "detail" in body
    bookmark_repo_mock.delete_by_id_for_user.assert_awaited_once_with("bm-user2-99", "user-1")


async def test_get_bookmarks_with_keyword_query_parameter() -> None:
    bookmark_repo_mock = MagicMock(spec=BookmarkRepository)
    bookmark_repo_mock.find_by_user_id = AsyncMock(return_value=[])
    app.dependency_overrides[get_bookmark_repository] = lambda: bookmark_repo_mock

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/rag/bookmarks?userId=user-1&keyword=검색어&page=2&limit=5")

    assert response.status_code == 200
    assert response.json() == []
    bookmark_repo_mock.find_by_user_id.assert_awaited_once_with("user-1", 2, 5, "검색어")


async def test_create_bookmark_turn_index_out_of_bounds_graceful() -> None:
    session_repo_mock = MagicMock(spec=ConversationSessionRepository)
    session_repo_mock.find_by_id_for_user = AsyncMock(return_value=_create_sample_session())
    app.dependency_overrides[get_conversation_session_repository] = lambda: session_repo_mock

    bookmark_repo_mock = MagicMock(spec=BookmarkRepository)
    bookmark_repo_mock.create = AsyncMock(return_value="bm-test-oob")
    app.dependency_overrides[get_bookmark_repository] = lambda: bookmark_repo_mock

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/rag/bookmarks?userId=user-1",
            json={"sessionId": "sess-1", "turnIndex": 999},
        )

    assert response.status_code in (200, 201)
    body = response.json()
    assert body["id"] == "bm-test-oob"
    assert body["turnIndex"] == 999
    assert body["content"] is None
    assert body["title"] == "파이프라인 질의"


async def test_create_bookmark_missing_session_id_fails_validation() -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/rag/bookmarks?userId=user-1",
            json={"turnIndex": 0},
        )

    assert response.status_code == 422


async def test_delete_bookmark_missing_user_id_fails_validation() -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.delete("/rag/bookmarks/bm-test-1")

    assert response.status_code == 422

