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


@pytest.mark.asyncio
async def test_수용기준1_평가_등록_후_삭제_호출_시_204_반환_및_평가_레코드_삭제() -> None:
    from tests.unit.feedback.fakes import (
        FakeAnswerFeedbackRepository,
        FakeSessionRepository,
    )

    feedback_repo = FakeAnswerFeedbackRepository()
    session_repo = FakeSessionRepository()
    session_repo.add("session-ac1", "user-1", ["user", "assistant"])

    real_service = FeedbackService(feedback_repo, session_repo)
    app.dependency_overrides[get_feedback_service] = lambda: real_service

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Given: 사용자가 특정 답변(turnIndex: 1)에 대해 이미 평가를 등록한 상태에서
        post_res = await client.post(
            "/rag/feedback",
            params={"userId": "user-1"},
            json={"sessionId": "session-ac1", "turnIndex": 1, "accuracy": 4, "helpfulness": 5},
        )
        assert post_res.status_code == 200

        get_res_before = await client.get(
            "/rag/feedback",
            params={"sessionId": "session-ac1", "userId": "user-1"},
        )
        assert len(get_res_before.json()) == 1

        # When: 해당 답변의 피드백 삭제 엔드포인트 호출
        delete_res = await client.delete(
            "/rag/feedback",
            params={"sessionId": "session-ac1", "turnIndex": 1, "userId": "user-1"},
        )

        # Then: 204 No Content 반환 및 저장된 평가가 삭제됨
        assert delete_res.status_code == 204
        assert delete_res.content == b""

        get_res_after = await client.get(
            "/rag/feedback",
            params={"sessionId": "session-ac1", "userId": "user-1"},
        )
        assert get_res_after.json() == []


@pytest.mark.asyncio
async def test_수용기준2_타인세션_또는_부정_turnIndex_삭제시_404_및_평가보존() -> None:
    from tests.unit.feedback.fakes import (
        FakeAnswerFeedbackRepository,
        FakeSessionRepository,
    )

    feedback_repo = FakeAnswerFeedbackRepository()
    session_repo = FakeSessionRepository()
    session_repo.add("session-owner", "user-owner", ["user", "assistant"])

    real_service = FeedbackService(feedback_repo, session_repo)
    app.dependency_overrides[get_feedback_service] = lambda: real_service

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Given: user-owner가 session-owner의 1번 턴에 평가를 등록한 상태
        post_res = await client.post(
            "/rag/feedback",
            params={"userId": "user-owner"},
            json={"sessionId": "session-owner", "turnIndex": 1, "accuracy": 4, "helpfulness": 5},
        )
        assert post_res.status_code == 200

        # When 1: 다른 사용자(user-attacker)가 삭제를 요청
        other_delete_res = await client.delete(
            "/rag/feedback",
            params={"sessionId": "session-owner", "turnIndex": 1, "userId": "user-attacker"},
        )
        # Then 1: 404 Not Found 반환
        assert other_delete_res.status_code == 404

        # When 2: 존재하지 않는 turnIndex(99 또는 -1)로 삭제를 요청
        out_of_bounds_res = await client.delete(
            "/rag/feedback",
            params={"sessionId": "session-owner", "turnIndex": 99, "userId": "user-owner"},
        )
        # Then 2: 404 Not Found 반환
        assert out_of_bounds_res.status_code == 404

        negative_turn_res = await client.delete(
            "/rag/feedback",
            params={"sessionId": "session-owner", "turnIndex": -1, "userId": "user-owner"},
        )
        assert negative_turn_res.status_code == 404

        # Then 3: 타인의 평가 레코드는 삭제되지 않고 유지되어야 함
        get_res = await client.get(
            "/rag/feedback",
            params={"sessionId": "session-owner", "userId": "user-owner"},
        )
        records = get_res.json()
        assert len(records) == 1
        assert records[0]["turnIndex"] == 1
        assert records[0]["accuracy"] == 4
        assert records[0]["helpfulness"] == 5


