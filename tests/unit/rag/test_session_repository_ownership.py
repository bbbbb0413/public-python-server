"""소유권 조건이 Mongo 쿼리 자체에 실려 나가는지 확인한다.

서비스에서 읽어 온 뒤 비교하는 방식은 호출부가 늘 때마다 빠질 수 있어
조건을 쿼리로 내렸다. 그 결정이 지켜지는지 여기서 고정한다.
"""

from unittest.mock import AsyncMock, MagicMock

from ai_service.rag.repository import ConversationSessionRepository


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
