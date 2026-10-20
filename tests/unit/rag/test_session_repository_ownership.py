"""소유권 조건이 Mongo 쿼리 자체에 실려 나가는지 확인한다.

서비스에서 읽어 온 뒤 비교하는 방식은 호출부가 늘 때마다 빠질 수 있어
조건을 쿼리로 내렸다. 그 결정이 지켜지는지 여기서 고정한다.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from ai_service.rag.repository import ConversationSessionRepository
from ai_service.rag.schemas import ConversationSession
from ai_service.rag.service import SessionNotFoundError, SessionService


def _make_repo() -> tuple[ConversationSessionRepository, MagicMock]:


    repo = ConversationSessionRepository.__new__(ConversationSessionRepository)
    collection = MagicMock()
    repo._collection = collection
    return repo, collection


async def test_조회가_세션id와_소유자를_함께_건다() -> None:
    repo, collection = _make_repo()
    collection.find_one = AsyncMock(return_value=None)

    await repo.find_by_id_for_user("session-1", "user-1")

    collection.find_one.assert_awaited_once_with(
        {"sessionId": "session-1", "userId": "user-1"}
    )


async def test_남의_세션은_없는_것으로_돌아온다() -> None:
    repo, collection = _make_repo()
    collection.find_one = AsyncMock(return_value=None)

    assert await repo.find_by_id_for_user("session-1", "침입자") is None


async def test_삭제가_세션id와_소유자를_함께_건다() -> None:
    repo, collection = _make_repo()
    collection.delete_one = AsyncMock(return_value=MagicMock(deleted_count=1))

    deleted = await repo.delete_by_id_for_user("session-1", "user-1")

    collection.delete_one.assert_awaited_once_with(
        {"sessionId": "session-1", "userId": "user-1"}
    )
    assert deleted is True


async def test_지운_것이_없으면_False() -> None:
    repo, collection = _make_repo()
    collection.delete_one = AsyncMock(return_value=MagicMock(deleted_count=0))

    assert await repo.delete_by_id_for_user("남의-세션", "침입자") is False


async def test_목록조회가_소유자조건을_건다() -> None:
    repo, collection = _make_repo()
    cursor_mock = MagicMock()
    cursor_mock.sort.return_value = cursor_mock
    cursor_mock.skip.return_value = cursor_mock
    cursor_mock.limit.return_value = cursor_mock
    cursor_mock.__aiter__.return_value = []
    collection.find.return_value = cursor_mock

    await repo.find_by_user_id("user-1", page=1, limit=10)

    collection.find.assert_called_once_with({"userId": "user-1"})
    cursor_mock.sort.assert_called_once_with("updatedAt", -1)
    cursor_mock.skip.assert_called_once_with(0)
    cursor_mock.limit.assert_called_once_with(10)


async def test_목록조회_키워드가_있으면_title_regex조건을_추가한다() -> None:
    repo, collection = _make_repo()
    cursor_mock = MagicMock()
    cursor_mock.sort.return_value = cursor_mock
    cursor_mock.skip.return_value = cursor_mock
    cursor_mock.limit.return_value = cursor_mock
    cursor_mock.__aiter__.return_value = []
    collection.find.return_value = cursor_mock

    await repo.find_by_user_id("user-1", page=2, limit=5, keyword=" 매출 (분석) ")

    collection.find.assert_called_once_with(
        {
            "userId": "user-1",
            "title": {"$regex": r"매출\ \(분석\)", "$options": "i"},
        }
    )
    cursor_mock.sort.assert_called_once_with("updatedAt", -1)
    cursor_mock.skip.assert_called_once_with(5)
    cursor_mock.limit.assert_called_once_with(5)


async def test_목록조회_키워드가_공백이면_title조건을_추가하지_않는다() -> None:
    repo, collection = _make_repo()
    cursor_mock = MagicMock()
    cursor_mock.sort.return_value = cursor_mock
    cursor_mock.skip.return_value = cursor_mock
    cursor_mock.limit.return_value = cursor_mock
    cursor_mock.__aiter__.return_value = []
    collection.find.return_value = cursor_mock

    await repo.find_by_user_id("user-1", page=1, limit=10, keyword="   ")

    collection.find.assert_called_once_with({"userId": "user-1"})


async def test_세션서비스_목록조회가_리포지토리에_키워드를_전달한다() -> None:
    repo = MagicMock(spec=ConversationSessionRepository)
    repo.find_by_user_id = AsyncMock(return_value=[])
    service = SessionService(repo)

    result = await service.get_sessions("user-1", page=1, limit=10, keyword="매출")

    assert result == []
    repo.find_by_user_id.assert_awaited_once_with("user-1", 1, 10, "매출")


async def test_업데이트가_turns와_updatedAt과_isBookmarked를_함께_갱신한다() -> None:
    repo, collection = _make_repo()
    collection.update_one = AsyncMock()
    session = ConversationSession.create("user-1", "질문").with_bookmark(True)

    await repo.update(session)

    collection.update_one.assert_awaited_once_with(
        {"sessionId": session.get_session_id()},
        {
            "$set": {
                "turns": [],
                "updatedAt": session.updated_at,
                "isBookmarked": True,
            }
        },
    )


async def test_세션서비스_북마크_변경을_소유자세션에만_적용하고_토글한다() -> None:
    repo = MagicMock(spec=ConversationSessionRepository)
    session = ConversationSession.create("user-1", "질문")
    repo.find_by_id_for_user = AsyncMock(return_value=session)
    repo.update = AsyncMock(side_effect=lambda s: s)
    service = SessionService(repo)

    toggled = await service.set_bookmark(session.get_session_id(), "user-1")

    assert toggled.is_bookmarked is True
    repo.find_by_id_for_user.assert_awaited_once_with(
        session.get_session_id(), "user-1"
    )
    repo.update.assert_awaited_once()


async def test_세션서비스_남의_세션_북마크_변경시_SessionNotFoundError() -> None:
    repo = MagicMock(spec=ConversationSessionRepository)
    repo.find_by_id_for_user = AsyncMock(return_value=None)
    repo.update = AsyncMock()
    service = SessionService(repo)

    with pytest.raises(SessionNotFoundError):
        await service.set_bookmark("other-session", "user-1", True)

    repo.update.assert_not_awaited()



