from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from bson import ObjectId

from ai_service.rag.repository import BookmarkRepository


def _make_repo() -> tuple[BookmarkRepository, MagicMock]:
    repo = BookmarkRepository.__new__(BookmarkRepository)
    collection = MagicMock()
    collection.find_one = AsyncMock(return_value=None)
    repo._collection = collection
    return repo, collection


class AsyncCursorMock:
    def __init__(self, items: list[dict]):
        self.items = items
        self._iter = iter(items)

    def sort(self, *args, **kwargs):
        return self

    def skip(self, *args, **kwargs):
        return self

    def limit(self, *args, **kwargs):
        return self

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration from None


@pytest.mark.asyncio
async def test_create_bookmark_returns_id() -> None:
    repo, collection = _make_repo()
    mock_id = ObjectId("507f1f77bcf86cd799439011")
    collection.insert_one = AsyncMock(return_value=MagicMock(inserted_id=mock_id))

    now = datetime.now(UTC)
    bookmark_dict = {
        "userId": "user-1",
        "sessionId": "sess-1",
        "turnIndex": 1,
        "title": "질의",
        "content": "답변",
        "note": "메모",
        "createdAt": now,
    }

    inserted_id = await repo.create(bookmark_dict)
    assert inserted_id == str(mock_id)
    collection.insert_one.assert_awaited_once_with(bookmark_dict)


@pytest.mark.asyncio
async def test_find_by_user_id_filters_and_paginates() -> None:
    repo, collection = _make_repo()
    now = datetime.now(UTC)
    raw_records = [
        {
            "_id": ObjectId("507f1f77bcf86cd799439011"),
            "userId": "user-1",
            "sessionId": "sess-1",
            "turnIndex": 1,
            "title": "지식베이스 질의",
            "content": "파이프라인 안내",
            "createdAt": now,
        }
    ]
    cursor_mock = AsyncCursorMock(raw_records)
    collection.find = MagicMock(return_value=cursor_mock)

    results = await repo.find_by_user_id(
        user_id="user-1", page=1, limit=10, query="파이프라인"
    )

    assert len(results) == 1
    assert results[0]["id"] == "507f1f77bcf86cd799439011"
    assert results[0]["content"] == "파이프라인 안내"
    collection.find.assert_called_once()
    called_query = collection.find.call_args[0][0]
    assert called_query["userId"] == "user-1"
    assert "$or" in called_query


@pytest.mark.asyncio
async def test_delete_by_id_for_user_success() -> None:
    repo, collection = _make_repo()
    collection.delete_one = AsyncMock(return_value=MagicMock(deleted_count=1))

    deleted = await repo.delete_by_id_for_user("507f1f77bcf86cd799439011", "user-1")

    assert deleted is True
    collection.delete_one.assert_awaited_once()


@pytest.mark.asyncio
async def test_delete_by_id_for_user_not_found_or_other_user() -> None:
    repo, collection = _make_repo()
    collection.delete_one = AsyncMock(return_value=MagicMock(deleted_count=0))

    deleted = await repo.delete_by_id_for_user("507f1f77bcf86cd799439011", "침입자")

    assert deleted is False


@pytest.mark.asyncio
async def test_find_by_session_and_user_queries_and_sorts_by_turn_index() -> None:
    repo, collection = _make_repo()
    now = datetime.now(UTC)
    raw_records = [
        {
            "_id": ObjectId("507f1f77bcf86cd799439011"),
            "userId": "user-1",
            "sessionId": "sess-1",
            "turnIndex": 1,
            "title": "지식베이스 질의",
            "content": "답변 1",
            "createdAt": now,
        },
        {
            "_id": ObjectId("507f1f77bcf86cd799439012"),
            "userId": "user-1",
            "sessionId": "sess-1",
            "turnIndex": 3,
            "title": "지식베이스 질의",
            "content": "답변 3",
            "createdAt": now,
        },
    ]
    cursor_mock = AsyncCursorMock(raw_records)
    collection.find = MagicMock(return_value=cursor_mock)

    results = await repo.find_by_session_and_user(session_id="sess-1", user_id="user-1")

    assert len(results) == 2
    assert results[0]["id"] == "507f1f77bcf86cd799439011"
    assert results[0]["turnIndex"] == 1
    assert results[1]["id"] == "507f1f77bcf86cd799439012"
    assert results[1]["turnIndex"] == 3

    collection.find.assert_called_once_with(
        {"sessionId": "sess-1", "userId": "user-1", "turnIndex": {"$ne": None}}
    )


@pytest.mark.asyncio
async def test_create_bookmark_existing_turn_upserts_and_returns_id() -> None:
    repo, collection = _make_repo()
    existing_id = ObjectId("507f1f77bcf86cd799439011")
    collection.find_one = AsyncMock(
        return_value={
            "_id": existing_id,
            "userId": "user-1",
            "sessionId": "sess-1",
            "turnIndex": 1,
            "title": "기존 제목",
            "content": "기존 답변",
            "note": "기존 메모",
        }
    )
    collection.update_one = AsyncMock(return_value=MagicMock(modified_count=1))
    collection.insert_one = AsyncMock()

    bookmark_dict = {
        "userId": "user-1",
        "sessionId": "sess-1",
        "turnIndex": 1,
        "title": "새 제목",
        "content": "새 답변",
        "note": "수정된 메모",
    }

    result_id = await repo.create(bookmark_dict)

    assert result_id == str(existing_id)
    collection.find_one.assert_awaited_once_with(
        {"sessionId": "sess-1", "userId": "user-1", "turnIndex": 1}
    )
    collection.update_one.assert_awaited_once()
    filter_arg, update_arg = collection.update_one.call_args[0]
    assert filter_arg == {"_id": existing_id}
    assert "$set" in update_arg
    set_fields = update_arg["$set"]
    assert set_fields["note"] == "수정된 메모"
    assert set_fields["content"] == "새 답변"
    assert set_fields["title"] == "새 제목"
    assert "updatedAt" in set_fields
    collection.insert_one.assert_not_called()


@pytest.mark.asyncio
async def test_create_bookmark_session_level_does_not_upsert() -> None:
    repo, collection = _make_repo()
    collection.find_one = AsyncMock()
    inserted_id = ObjectId("507f1f77bcf86cd799439099")
    collection.insert_one = AsyncMock(return_value=MagicMock(inserted_id=inserted_id))

    bookmark_dict = {
        "userId": "user-1",
        "sessionId": "sess-1",
        "turnIndex": None,
        "title": "세션 북마크",
    }

    result_id = await repo.create(bookmark_dict)

    assert result_id == str(inserted_id)
    collection.find_one.assert_not_called()
    collection.insert_one.assert_awaited_once()


