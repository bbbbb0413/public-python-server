from unittest.mock import AsyncMock, MagicMock

import pytest

from ai_service.feedback.repository import AnswerFeedbackRepository


def _make_repo() -> tuple[AnswerFeedbackRepository, MagicMock]:
    repo = AnswerFeedbackRepository.__new__(AnswerFeedbackRepository)
    mock_collection = MagicMock()
    repo._collection = mock_collection
    return repo, mock_collection


@pytest.mark.asyncio
async def test_delete_feedback_success() -> None:
    repo, collection = _make_repo()
    result_mock = MagicMock()
    result_mock.deleted_count = 1
    collection.delete_one = AsyncMock(return_value=result_mock)

    deleted = await repo.delete(session_id="session-1", turn_index=1, user_id="user-1")

    assert deleted is True
    collection.delete_one.assert_awaited_once_with(
        {"sessionId": "session-1", "turnIndex": 1, "userId": "user-1"}
    )


@pytest.mark.asyncio
async def test_delete_feedback_when_not_found_returns_false() -> None:
    repo, collection = _make_repo()
    result_mock = MagicMock()
    result_mock.deleted_count = 0
    collection.delete_one = AsyncMock(return_value=result_mock)

    deleted = await repo.delete(session_id="session-1", turn_index=1, user_id="user-1")

    assert deleted is False
    collection.delete_one.assert_awaited_once_with(
        {"sessionId": "session-1", "turnIndex": 1, "userId": "user-1"}
    )
