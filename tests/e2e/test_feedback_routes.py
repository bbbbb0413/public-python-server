from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from ai_service.feedback.dependencies import get_feedback_service
from ai_service.feedback.service import FeedbackError, FeedbackService
from ai_service.main import app


@pytest.fixture(autouse=True)
def _clear_overrides():
    yield
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_delete_feedback_success_returns_204() -> None:
    service_mock = MagicMock(spec=FeedbackService)
    service_mock.delete = AsyncMock(return_value=None)
    app.dependency_overrides[get_feedback_service] = lambda: service_mock

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.delete(
            "/rag/feedback",
            params={"sessionId": "session-1", "turnIndex": 1, "userId": "user-1"},
        )

    assert response.status_code == 204
    assert response.content == b""
    service_mock.delete.assert_awaited_once_with("session-1", 1, "user-1")


@pytest.mark.asyncio
async def test_delete_feedback_session_not_found_returns_404() -> None:
    service_mock = MagicMock(spec=FeedbackService)
    service_mock.delete = AsyncMock(side_effect=FeedbackError("대상 세션을 찾을 수 없습니다.", 404))
    app.dependency_overrides[get_feedback_service] = lambda: service_mock

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.delete(
            "/rag/feedback",
            params={"sessionId": "non-existent", "turnIndex": 1, "userId": "user-1"},
        )

    assert response.status_code == 404
    assert response.json()["detail"] == "대상 세션을 찾을 수 없습니다."


@pytest.mark.asyncio
async def test_delete_feedback_invalid_turn_role_returns_400() -> None:
    service_mock = MagicMock(spec=FeedbackService)
    service_mock.delete = AsyncMock(
        side_effect=FeedbackError("답변이 아닌 턴에는 평가를 남길 수 없습니다.", 400)
    )
    app.dependency_overrides[get_feedback_service] = lambda: service_mock

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.delete(
            "/rag/feedback",
            params={"sessionId": "session-1", "turnIndex": 0, "userId": "user-1"},
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "답변이 아닌 턴에는 평가를 남길 수 없습니다."



@pytest.mark.asyncio
async def test_delete_feedback_missing_required_params_returns_422() -> None:
    service_mock = MagicMock(spec=FeedbackService)
    app.dependency_overrides[get_feedback_service] = lambda: service_mock

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.delete("/rag/feedback?sessionId=session-1")

    assert response.status_code == 422

