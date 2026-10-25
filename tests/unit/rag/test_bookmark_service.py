from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from ai_service.rag.repository import BookmarkRepository, ConversationSessionRepository
from ai_service.rag.schemas import (
    ConversationSession,
    CreateBookmarkIn,
    RestoreProps,
    TurnRecord,
)
from ai_service.rag.service import BookmarkNotFoundError, BookmarkService


def _create_sample_session(
    session_id: str = "sess-1", user_id: str = "user-1"
) -> ConversationSession:
    now = datetime.now(UTC)
    props = RestoreProps(
        session_id=session_id,
        user_id=user_id,
        title="세션 제목: 파이프라인 가이드",
        turns=[
            TurnRecord(role="user", content="파이프라인이 뭔가요?", created_at=now),
            TurnRecord(role="assistant", content="RAG 검색 결과입니다.", created_at=now),
        ],
        created_at=now,
        updated_at=now,
    )
    return ConversationSession.restore(props)


@pytest.mark.asyncio
async def test_create_bookmark_with_turn_index() -> None:
    session_repo = MagicMock(spec=ConversationSessionRepository)
    session_repo.find_by_id_for_user = AsyncMock(return_value=_create_sample_session())

    bookmark_repo = MagicMock(spec=BookmarkRepository)
    bookmark_repo.create = AsyncMock(return_value="bm-new-1")

    service = BookmarkService(bookmark_repo=bookmark_repo, session_repo=session_repo)

    dto = CreateBookmarkIn(sessionId="sess-1", turnIndex=1, note="노트")
    result = await service.create_bookmark(user_id="user-1", dto=dto)

    assert result.id == "bm-new-1"
    assert result.session_id == "sess-1"
    assert result.turn_index == 1
    assert result.content == "RAG 검색 결과입니다."
    assert result.title == "세션 제목: 파이프라인 가이드"
    assert result.note == "노트"

    bookmark_repo.create.assert_awaited_once()
    saved_doc = bookmark_repo.create.call_args[0][0]
    assert saved_doc["userId"] == "user-1"
    assert saved_doc["content"] == "RAG 검색 결과입니다."


@pytest.mark.asyncio
async def test_create_bookmark_without_turn_index_saves_session_level() -> None:
    session_repo = MagicMock(spec=ConversationSessionRepository)
    session_repo.find_by_id_for_user = AsyncMock(return_value=_create_sample_session())

    bookmark_repo = MagicMock(spec=BookmarkRepository)
    bookmark_repo.create = AsyncMock(return_value="bm-new-2")

    service = BookmarkService(bookmark_repo=bookmark_repo, session_repo=session_repo)

    dto = CreateBookmarkIn(sessionId="sess-1", turnIndex=None)
    result = await service.create_bookmark(user_id="user-1", dto=dto)

    assert result.id == "bm-new-2"
    assert result.turn_index is None
    assert result.content is None
    assert result.title == "세션 제목: 파이프라인 가이드"


@pytest.mark.asyncio
async def test_create_bookmark_when_session_not_found_handles_gracefully() -> None:
    session_repo = MagicMock(spec=ConversationSessionRepository)
    session_repo.find_by_id_for_user = AsyncMock(return_value=None)

    bookmark_repo = MagicMock(spec=BookmarkRepository)
    bookmark_repo.create = AsyncMock(return_value="bm-new-3")

    service = BookmarkService(bookmark_repo=bookmark_repo, session_repo=session_repo)

    dto = CreateBookmarkIn(sessionId="sess-nonexistent", turnIndex=0)
    result = await service.create_bookmark(user_id="user-1", dto=dto)

    assert result.id == "bm-new-3"
    assert result.session_id == "sess-nonexistent"
    assert result.content is None
    assert "sess-nonexistent" in (result.title or "")


@pytest.mark.asyncio
async def test_list_bookmarks_returns_mapped_items() -> None:
    now = datetime.now(UTC)
    bookmark_repo = MagicMock(spec=BookmarkRepository)
    bookmark_repo.find_by_user_id = AsyncMock(
        return_value=[
            {
                "id": "bm-1",
                "sessionId": "sess-1",
                "turnIndex": 1,
                "title": "지식베이스 질의",
                "content": "파이프라인 안내",
                "note": None,
                "createdAt": now,
            }
        ]
    )
    session_repo = MagicMock(spec=ConversationSessionRepository)
    service = BookmarkService(bookmark_repo=bookmark_repo, session_repo=session_repo)

    items = await service.list_bookmarks(user_id="user-1", page=1, limit=10, query="파이프라인")

    assert len(items) == 1
    assert items[0].id == "bm-1"
    assert items[0].content == "파이프라인 안내"
    bookmark_repo.find_by_user_id.assert_awaited_once_with("user-1", 1, 10, "파이프라인")


@pytest.mark.asyncio
async def test_delete_bookmark_success_and_failure() -> None:
    bookmark_repo = MagicMock(spec=BookmarkRepository)
    bookmark_repo.delete_by_id_for_user = AsyncMock(side_effect=[True, False])
    session_repo = MagicMock(spec=ConversationSessionRepository)
    service = BookmarkService(bookmark_repo=bookmark_repo, session_repo=session_repo)

    # Success
    await service.delete_bookmark("bm-1", "user-1")

    # Failure (not found or not owner)
    with pytest.raises(BookmarkNotFoundError):
        await service.delete_bookmark("bm-2", "user-1")


@pytest.mark.asyncio
async def test_create_bookmark_with_negative_or_out_of_range_turn_index() -> None:
    session_repo = MagicMock(spec=ConversationSessionRepository)
    session_repo.find_by_id_for_user = AsyncMock(return_value=_create_sample_session())

    bookmark_repo = MagicMock(spec=BookmarkRepository)
    bookmark_repo.create = AsyncMock(return_value="bm-oob-1")

    service = BookmarkService(bookmark_repo=bookmark_repo, session_repo=session_repo)

    # Negative index
    dto_neg = CreateBookmarkIn(sessionId="sess-1", turnIndex=-1)
    res_neg = await service.create_bookmark(user_id="user-1", dto=dto_neg)
    assert res_neg.turn_index == -1
    assert res_neg.content is None
    assert res_neg.title == "세션 제목: 파이프라인 가이드"

    # Out of range positive index
    dto_pos = CreateBookmarkIn(sessionId="sess-1", turnIndex=100)
    res_pos = await service.create_bookmark(user_id="user-1", dto=dto_pos)
    assert res_pos.turn_index == 100
    assert res_pos.content is None

