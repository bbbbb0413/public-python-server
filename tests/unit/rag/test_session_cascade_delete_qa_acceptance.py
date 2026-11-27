from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from ai_service.feedback.dependencies import get_answer_feedback_repository
from ai_service.feedback.repository import AnswerFeedbackRepository
from ai_service.main import app
from ai_service.rag.dependencies import (
    get_bookmark_repository,
    get_conversation_session_repository,
)
from ai_service.rag.repository import BookmarkRepository, ConversationSessionRepository
from ai_service.rag.service import SessionNotFoundError, SessionService


@pytest.fixture
def session_qa_fixture():
    session_repo = MagicMock(spec=ConversationSessionRepository)
    bookmark_repo = MagicMock(spec=BookmarkRepository)
    feedback_repo = MagicMock(spec=AnswerFeedbackRepository)

    service = SessionService(
        repo=session_repo,
        bookmark_repo=bookmark_repo,
        feedback_repo=feedback_repo,
    )

    return {
        "service": service,
        "session_repo": session_repo,
        "bookmark_repo": bookmark_repo,
        "feedback_repo": feedback_repo,
    }


@pytest.mark.asyncio
async def test_acceptance_criteria_1_cascade_delete_success(session_qa_fixture) -> None:
    """수용 기준 1 검증:
    Given 사용자가 북마크와 답변 피드백이 등록된 대화 세션을 보유하고 있을 때
    When 사용자가 해당 대화 세션을 삭제하면
    Then conversation_sessions 컬렉션에서 세션이 삭제되고,
         연관된 bookmarks와 feedbacks 도큐먼트도 함께 데이터베이스에서 삭제된다.
    """
    service = session_qa_fixture["service"]
    session_repo = session_qa_fixture["session_repo"]
    bookmark_repo = session_qa_fixture["bookmark_repo"]
    feedback_repo = session_qa_fixture["feedback_repo"]

    session_repo.delete_by_id_for_user = AsyncMock(return_value=True)
    bookmark_repo.delete_by_session_for_user = AsyncMock(return_value=2)
    feedback_repo.delete_by_session_for_user = AsyncMock(return_value=3)

    session_id = "session-with-bookmarks-and-feedback"
    user_id = "user-owner"

    await service.delete_session(session_id=session_id, user_id=user_id)

    session_repo.delete_by_id_for_user.assert_awaited_once_with(session_id, user_id)
    bookmark_repo.delete_by_session_for_user.assert_awaited_once_with(session_id, user_id)
    feedback_repo.delete_by_session_for_user.assert_awaited_once_with(session_id, user_id)


@pytest.mark.asyncio
async def test_acceptance_criteria_1_cascade_delete_via_e2e_route() -> None:
    """수용 기준 1 E2E API 라우트 검증:
    Given 사용자가 북마크와 답변 피드백이 등록된 세션을 보유하고 있을 때
    When DELETE /rag/sessions/{session_id}?userId={user_id} 호출 시
    Then 204 No Content가 반환되고 세션, 북마크, 피드백 저장소 모두 삭제 메서드가 호출된다.
    """
    session_repo_mock = MagicMock(spec=ConversationSessionRepository)
    session_repo_mock.delete_by_id_for_user = AsyncMock(return_value=True)

    bookmark_repo_mock = MagicMock(spec=BookmarkRepository)
    bookmark_repo_mock.delete_by_session_for_user = AsyncMock(return_value=2)

    feedback_repo_mock = MagicMock(spec=AnswerFeedbackRepository)
    feedback_repo_mock.delete_by_session_for_user = AsyncMock(return_value=1)

    app.dependency_overrides[get_conversation_session_repository] = lambda: session_repo_mock
    app.dependency_overrides[get_bookmark_repository] = lambda: bookmark_repo_mock
    app.dependency_overrides[get_answer_feedback_repository] = lambda: feedback_repo_mock

    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.delete("/rag/sessions/session-e2e-ac1?userId=user-e2e-1")

        assert response.status_code == 204
        session_repo_mock.delete_by_id_for_user.assert_awaited_once_with(
            "session-e2e-ac1", "user-e2e-1"
        )
        bookmark_repo_mock.delete_by_session_for_user.assert_awaited_once_with(
            "session-e2e-ac1", "user-e2e-1"
        )
        feedback_repo_mock.delete_by_session_for_user.assert_awaited_once_with(
            "session-e2e-ac1", "user-e2e-1"
        )
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_acceptance_criteria_3_delete_session_without_bookmarks_or_feedback(
    session_qa_fixture,
) -> None:
    """수용 기준 3 검증 (경계 케이스):
    Given 북마크나 피드백이 하나도 없는 대화 세션이 존재할 때
    When 사용자가 해당 세션을 삭제하면
    Then 오류 없이 세션이 정상 삭제되고 204 No Content가 반환된다.
    """
    service = session_qa_fixture["service"]
    session_repo = session_qa_fixture["session_repo"]
    bookmark_repo = session_qa_fixture["bookmark_repo"]
    feedback_repo = session_qa_fixture["feedback_repo"]

    session_repo.delete_by_id_for_user = AsyncMock(return_value=True)
    bookmark_repo.delete_by_session_for_user = AsyncMock(return_value=0)
    feedback_repo.delete_by_session_for_user = AsyncMock(return_value=0)

    session_id = "session-empty-bm-feedback"
    user_id = "user-123"

    # 예외 없이 정상 종료되어야 함
    await service.delete_session(session_id=session_id, user_id=user_id)

    session_repo.delete_by_id_for_user.assert_awaited_once_with(session_id, user_id)
    bookmark_repo.delete_by_session_for_user.assert_awaited_once_with(session_id, user_id)
    feedback_repo.delete_by_session_for_user.assert_awaited_once_with(session_id, user_id)


@pytest.mark.asyncio
async def test_acceptance_criteria_3_delete_session_without_optional_dependencies() -> None:
    """수용 기준 3 보완 검증:
    선택적 의존성인 bookmark_repo 또는 feedback_repo가 None인 상태에서도
    오류 없이 세션 단독 삭제가 안전하게 수행된다.
    """
    session_repo = MagicMock(spec=ConversationSessionRepository)
    session_repo.delete_by_id_for_user = AsyncMock(return_value=True)

    service = SessionService(repo=session_repo, bookmark_repo=None, feedback_repo=None)
    await service.delete_session(session_id="session-pure", user_id="user-pure")

    session_repo.delete_by_id_for_user.assert_awaited_once_with("session-pure", "user-pure")


@pytest.mark.asyncio
async def test_acceptance_criteria_4_missing_or_other_user_session_raises_404_and_no_cascade(
    session_qa_fixture,
) -> None:
    """수용 기준 4 검증 (경계 케이스):
    Given 존재하지 않는 세션 ID이거나 다른 사용자가 소유한 세션 ID로 삭제를 요청했을 때
    When DELETE /rag/sessions/{session_id}?userId={user_id}를 호출하면
    Then 404 SessionNotFoundError가 발생하고 타인의 세션, 북마크, 피드백은 삭제되지 않는다.
    """
    service = session_qa_fixture["service"]
    session_repo = session_qa_fixture["session_repo"]
    bookmark_repo = session_qa_fixture["bookmark_repo"]
    feedback_repo = session_qa_fixture["feedback_repo"]

    session_repo.delete_by_id_for_user = AsyncMock(return_value=False)
    bookmark_repo.delete_by_session_for_user = AsyncMock()
    feedback_repo.delete_by_session_for_user = AsyncMock()

    other_user_session_id = "session-owned-by-victim"
    attacker_user_id = "attacker-user"

    with pytest.raises(SessionNotFoundError):
        await service.delete_session(
            session_id=other_user_session_id, user_id=attacker_user_id
        )

    session_repo.delete_by_id_for_user.assert_awaited_once_with(
        other_user_session_id, attacker_user_id
    )
    bookmark_repo.delete_by_session_for_user.assert_not_awaited()
    feedback_repo.delete_by_session_for_user.assert_not_awaited()


@pytest.mark.asyncio
async def test_acceptance_criteria_4_e2e_route_other_user_session_returns_404() -> None:
    """수용 기준 4 E2E API 라우트 검증:
    다른 사용자의 세션 삭제 시 404 반환 및 연쇄 삭제 방지 확인.
    """
    session_repo_mock = MagicMock(spec=ConversationSessionRepository)
    session_repo_mock.delete_by_id_for_user = AsyncMock(return_value=False)

    bookmark_repo_mock = MagicMock(spec=BookmarkRepository)
    bookmark_repo_mock.delete_by_session_for_user = AsyncMock()

    feedback_repo_mock = MagicMock(spec=AnswerFeedbackRepository)
    feedback_repo_mock.delete_by_session_for_user = AsyncMock()

    app.dependency_overrides[get_conversation_session_repository] = lambda: session_repo_mock
    app.dependency_overrides[get_bookmark_repository] = lambda: bookmark_repo_mock
    app.dependency_overrides[get_answer_feedback_repository] = lambda: feedback_repo_mock

    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.delete(
                "/rag/sessions/victim-session-id?userId=attacker-id"
            )

        assert response.status_code == 404
        assert response.json()["detail"] == "대화 세션을 찾을 수 없습니다."
        session_repo_mock.delete_by_id_for_user.assert_awaited_once_with(
            "victim-session-id", "attacker-id"
        )
        bookmark_repo_mock.delete_by_session_for_user.assert_not_awaited()
        feedback_repo_mock.delete_by_session_for_user.assert_not_awaited()
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_repository_user_isolation_filter_criteria() -> None:
    """사용자 격리 필터 검증:
    BookmarkRepository와 AnswerFeedbackRepository의 delete_by_session_for_user는
    반드시 sessionId와 userId 조건을 모두 포함하여 delete_many를 실행해야 한다.
    """
    bookmark_col = MagicMock()
    bm_res = MagicMock()
    bm_res.deleted_count = 5
    bookmark_col.delete_many = AsyncMock(return_value=bm_res)

    db_mock = MagicMock()
    db_mock.__getitem__.return_value = bookmark_col

    bm_repo = BookmarkRepository(db_mock)
    bm_count = await bm_repo.delete_by_session_for_user("sess-target", "user-isolated")

    assert bm_count == 5
    bookmark_col.delete_many.assert_awaited_once_with(
        {"sessionId": "sess-target", "userId": "user-isolated"}
    )

    feedback_col = MagicMock()
    fb_res = MagicMock()
    fb_res.deleted_count = 3
    feedback_col.delete_many = AsyncMock(return_value=fb_res)
    db_mock.__getitem__.return_value = feedback_col

    fb_repo = AnswerFeedbackRepository(db_mock)
    fb_count = await fb_repo.delete_by_session_for_user("sess-target", "user-isolated")

    assert fb_count == 3
    feedback_col.delete_many.assert_awaited_once_with(
        {"sessionId": "sess-target", "userId": "user-isolated"}
    )
