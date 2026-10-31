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
                iterations=2,
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
    assert body["turns"][0]["iterations"] is None

    assert body["turns"][1]["role"] == "assistant"
    assert body["turns"][1]["content"] == "환불 가능합니다."
    assert body["turns"][1]["sources"] == sources
    assert body["turns"][1]["confidence"] == 0.92
    assert body["turns"][1]["missing"] == ["예외 규정"]
    assert body["turns"][1]["iterations"] == 2


async def test_get_session_returns_turns_with_refined_queries() -> None:
    now = datetime.now(UTC)
    props = RestoreProps(
        session_id="session-refined-123",
        user_id="user-1",
        title="정제 쿼리 세션",
        turns=[
            TurnRecord(role="user", content="복합 질문", created_at=now),
            TurnRecord(
                role="assistant",
                content="종합 답변입니다.",
                created_at=now,
                confidence=0.88,
                missing=[],
                iterations=2,
                refined_queries=["보완 검색 쿼리 1", "보완 검색 쿼리 2"],
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
        response = await client.get("/rag/sessions/session-refined-123?userId=user-1")

    assert response.status_code == 200
    body = response.json()
    assert body["sessionId"] == "session-refined-123"
    assert len(body["turns"]) == 2
    assert body["turns"][0]["refinedQueries"] is None
    assert body["turns"][1]["refinedQueries"] == ["보완 검색 쿼리 1", "보완 검색 쿼리 2"]


async def test_get_session_returns_turns_with_empty_refined_queries() -> None:
    now = datetime.now(UTC)
    props = RestoreProps(
        session_id="session-empty-refined-123",
        user_id="user-1",
        title="1회차 성공 세션",
        turns=[
            TurnRecord(role="user", content="단순/1회차 질문", created_at=now),
            TurnRecord(
                role="assistant",
                content="1회차 만족 답변입니다.",
                created_at=now,
                confidence=0.95,
                missing=[],
                iterations=1,
                refined_queries=[],
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
        response = await client.get("/rag/sessions/session-empty-refined-123?userId=user-1")

    assert response.status_code == 200
    body = response.json()
    assert body["sessionId"] == "session-empty-refined-123"
    assert len(body["turns"]) == 2
    assert body["turns"][0]["refinedQueries"] is None
    assert body["turns"][1]["refinedQueries"] == []


async def test_get_session_with_zero_confidence_and_ungrounded_warning() -> None:
    now = datetime.now(UTC)
    props = RestoreProps(
        session_id="session-zero-conf-123",
        user_id="user-1",
        title="근거 없는 답변 세션",
        turns=[
            TurnRecord(role="user", content="관련 없는 질문", created_at=now),
            TurnRecord(
                role="assistant",
                content="일반 지식 기반 답변입니다.",
                created_at=now,
                sources=None,
                confidence=0.0,
                missing=["관련 지식베이스 문서를 찾지 못함"],
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
        response = await client.get("/rag/sessions/session-zero-conf-123?userId=user-1")

    assert response.status_code == 200
    body = response.json()
    assert body["sessionId"] == "session-zero-conf-123"
    assert len(body["turns"]) == 2
    assert body["turns"][1]["role"] == "assistant"
    assert body["turns"][1]["content"] == "일반 지식 기반 답변입니다."
    assert body["turns"][1]["sources"] is None
    assert body["turns"][1]["confidence"] == 0.0
    assert body["turns"][1]["missing"] == ["관련 지식베이스 문서를 찾지 못함"]
    assert body["turns"][1]["iterations"] is None


async def test_get_session_legacy_data_returns_none_metadata() -> None:
    now = datetime.now(UTC)
    legacy_sources = [
        {"fileName": "legacy.pdf", "chunkIndex": 0, "documentId": "doc-old", "snippet": "과거 문서"}
    ]
    legacy_mongo_doc: dict[str, Any] = {
        "sessionId": "session-legacy-123",
        "userId": "user-1",
        "title": "레거시 세션",
        "turns": [
            {"role": "user", "content": "질문", "createdAt": now},
            {
                "role": "assistant",
                "content": "답변",
                "createdAt": now,
                "sources": legacy_sources,
                "confidence": 0.87,
                "missing": [],
            },
            {"role": "assistant", "content": "초기 레거시 답변", "createdAt": now},
        ],
        "createdAt": now,
        "updatedAt": now,
    }
    session = ConversationSessionRepository._to_domain(legacy_mongo_doc)

    repo_mock = MagicMock(spec=ConversationSessionRepository)
    repo_mock.find_by_id_for_user = AsyncMock(return_value=session)
    app.dependency_overrides[get_conversation_session_repository] = lambda: repo_mock

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/rag/sessions/session-legacy-123?userId=user-1")

    assert response.status_code == 200
    body = response.json()
    assert body["sessionId"] == "session-legacy-123"
    assert len(body["turns"]) == 3
    assert body["turns"][0]["iterations"] is None
    assert body["turns"][1]["content"] == "답변"
    assert body["turns"][1]["sources"] == legacy_sources
    assert body["turns"][1]["confidence"] == 0.87
    assert body["turns"][1]["missing"] == []
    assert body["turns"][1]["iterations"] is None
    assert body["turns"][2]["sources"] is None
    assert body["turns"][2]["confidence"] is None
    assert body["turns"][2]["missing"] is None
    assert body["turns"][2]["iterations"] is None


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


async def test_get_sessions_list_passes_keyword_and_pagination() -> None:
    now = datetime.now(UTC)
    session = ConversationSession.restore(
        RestoreProps(
            session_id="session-1",
            user_id="user-1",
            title="매출 보고서 분석",
            turns=[TurnRecord(role="user", content="질문", created_at=now)],
            created_at=now,
            updated_at=now,
        )
    )
    repo_mock = MagicMock(spec=ConversationSessionRepository)
    repo_mock.find_by_user_id = AsyncMock(return_value=[session])
    app.dependency_overrides[get_conversation_session_repository] = lambda: repo_mock

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/rag/sessions?userId=user-1&page=2&limit=5&keyword=매출")

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["sessionId"] == "session-1"
    assert body[0]["title"] == "매출 보고서 분석"
    repo_mock.find_by_user_id.assert_awaited_once_with("user-1", 2, 5, "매출")


async def test_get_sessions_list_empty_result_returns_empty_list() -> None:
    repo_mock = MagicMock(spec=ConversationSessionRepository)
    repo_mock.find_by_user_id = AsyncMock(return_value=[])
    app.dependency_overrides[get_conversation_session_repository] = lambda: repo_mock

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(
            "/rag/sessions?userId=user-1&keyword=존재하지않는제목"
        )

    assert response.status_code == 200
    assert response.json() == []
    repo_mock.find_by_user_id.assert_awaited_once_with("user-1", 1, 10, "존재하지않는제목")


async def test_get_sessions_list_passes_q_param_and_returns_is_bookmarked() -> None:
    now = datetime.now(UTC)
    session = ConversationSession.restore(
        RestoreProps(
            session_id="session-1",
            user_id="user-1",
            title="매출 보고서 분석",
            turns=[TurnRecord(role="user", content="질문", created_at=now)],
            created_at=now,
            updated_at=now,
            is_bookmarked=True,
        )
    )
    repo_mock = MagicMock(spec=ConversationSessionRepository)
    repo_mock.find_by_user_id = AsyncMock(return_value=[session])
    app.dependency_overrides[get_conversation_session_repository] = lambda: repo_mock

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/rag/sessions?userId=user-1&page=1&limit=50&q=매출")

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["sessionId"] == "session-1"
    assert body[0]["title"] == "매출 보고서 분석"
    assert body[0]["isBookmarked"] is True
    repo_mock.find_by_user_id.assert_awaited_once_with("user-1", 1, 50, "매출")


async def test_patch_session_bookmark_sets_explicit_and_toggles() -> None:
    now = datetime.now(UTC)
    current = ConversationSession.restore(
        RestoreProps(
            session_id="session-1",
            user_id="user-1",
            title="북마크 테스트",
            turns=[],
            created_at=now,
            updated_at=now,
            is_bookmarked=False,
        )
    )
    repo_mock = MagicMock(spec=ConversationSessionRepository)

    async def _find_current(_session_id: str, _user_id: str) -> ConversationSession:
        return current

    async def _save_updated(updated: ConversationSession) -> ConversationSession:
        nonlocal current
        current = updated
        return updated

    repo_mock.find_by_id_for_user = AsyncMock(side_effect=_find_current)
    repo_mock.update = AsyncMock(side_effect=_save_updated)
    app.dependency_overrides[get_conversation_session_repository] = lambda: repo_mock

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res_explicit = await client.patch(
            "/rag/sessions/session-1/bookmark?userId=user-1",
            json={"bookmarked": True},
        )
        res_toggle_off = await client.patch(
            "/rag/sessions/session-1/bookmark?userId=user-1",
        )
        res_toggle_on = await client.patch(
            "/rag/sessions/session-1/bookmark?userId=user-1",
        )

    assert res_explicit.status_code == 200
    assert res_explicit.json() == {"sessionId": "session-1", "bookmarked": True}
    assert res_toggle_off.status_code == 200
    assert res_toggle_off.json() == {"sessionId": "session-1", "bookmarked": False}
    assert res_toggle_on.status_code == 200
    assert res_toggle_on.json() == {"sessionId": "session-1", "bookmarked": True}


async def test_get_sessions_list_prioritizes_keyword_over_q_and_handles_legacy_doc() -> None:
    now = datetime.now(UTC)
    legacy_mongo_doc: dict[str, Any] = {
        "sessionId": "session-legacy-1",
        "userId": "user-1",
        "title": "매출 레거시 문서",
        "turns": [{"role": "user", "content": "질문", "createdAt": now}],
        "createdAt": now,
        "updatedAt": now,
    }
    legacy_session = ConversationSessionRepository._to_domain(legacy_mongo_doc)
    repo_mock = MagicMock(spec=ConversationSessionRepository)
    repo_mock.find_by_user_id = AsyncMock(return_value=[legacy_session])
    app.dependency_overrides[get_conversation_session_repository] = lambda: repo_mock

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(
            "/rag/sessions?userId=user-1&keyword=우선키워드&q=후순위키워드"
        )

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["sessionId"] == "session-legacy-1"
    assert body[0]["isBookmarked"] is False
    repo_mock.find_by_user_id.assert_awaited_once_with("user-1", 1, 10, "우선키워드")


async def test_bookmark_routes_require_user_id() -> None:
    repo_mock = MagicMock(spec=ConversationSessionRepository)
    app.dependency_overrides[get_conversation_session_repository] = lambda: repo_mock

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        patch_res = await client.patch("/rag/sessions/session-1/bookmark")
        post_res = await client.post("/rag/sessions/session-1/bookmark")
        delete_res = await client.delete("/rag/sessions/session-1/bookmark")

    assert patch_res.status_code == 422
    assert post_res.status_code == 422
    assert delete_res.status_code == 422



async def test_post_and_delete_session_bookmark() -> None:
    now = datetime.now(UTC)
    session = ConversationSession.restore(
        RestoreProps(
            session_id="session-1",
            user_id="user-1",
            title="북마크 등록/해제",
            turns=[],
            created_at=now,
            updated_at=now,
            is_bookmarked=False,
        )
    )
    repo_mock = MagicMock(spec=ConversationSessionRepository)
    repo_mock.find_by_id_for_user = AsyncMock(return_value=session)
    repo_mock.update = AsyncMock(side_effect=lambda s: s)
    app.dependency_overrides[get_conversation_session_repository] = lambda: repo_mock

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        post_res = await client.post("/rag/sessions/session-1/bookmark?userId=user-1")
        delete_res = await client.delete("/rag/sessions/session-1/bookmark?userId=user-1")

    assert post_res.status_code == 200
    assert post_res.json() == {"sessionId": "session-1", "bookmarked": True}
    assert delete_res.status_code == 204


async def test_bookmark_routes_return_indistinguishable_404_for_other_or_missing_session() -> None:
    repo_mock = MagicMock(spec=ConversationSessionRepository)
    repo_mock.find_by_id_for_user = AsyncMock(return_value=None)
    repo_mock.update = AsyncMock()
    app.dependency_overrides[get_conversation_session_repository] = lambda: repo_mock

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        patch_res = await client.patch(
            "/rag/sessions/other-session/bookmark?userId=user-1",
            json={"bookmarked": True},
        )
        post_res = await client.post("/rag/sessions/other-session/bookmark?userId=user-1")
        delete_res = await client.delete("/rag/sessions/missing-session/bookmark?userId=user-1")

    assert patch_res.status_code == 404
    assert post_res.status_code == 404
    assert delete_res.status_code == 404
    assert patch_res.json() == {"detail": "대화 세션을 찾을 수 없습니다."}
    assert post_res.json() == patch_res.json()
    assert delete_res.json() == patch_res.json()
    repo_mock.update.assert_not_awaited()


async def test_get_session_returns_is_cancelled_field() -> None:
    now = datetime.now(UTC)
    props = RestoreProps(
        session_id="session-cancel-detail",
        user_id="user-1",
        title="중단된 세션 상세",
        turns=[
            TurnRecord(role="user", content="질문입니다", created_at=now),
            TurnRecord(
                role="assistant",
                content="답변 생성이 중단되었습니다.",
                created_at=now,
                is_cancelled=True,
            ),
            TurnRecord(
                role="assistant",
                content="기존 레거시 완료 답변",
                created_at=now,
                is_cancelled=False,
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
        response = await client.get("/rag/sessions/session-cancel-detail?userId=user-1")

    assert response.status_code == 200
    body = response.json()
    assert body["sessionId"] == "session-cancel-detail"
    assert len(body["turns"]) == 3
    assert body["turns"][0]["isCancelled"] is False
    assert body["turns"][1]["isCancelled"] is True
    assert body["turns"][1]["content"] == "답변 생성이 중단되었습니다."
    assert body["turns"][2]["isCancelled"] is False


async def test_get_session_detail_handles_legacy_mongodb_document_without_iscancelled() -> None:
    """수용 기준 1: Given 중단 플래그가 없는 기존 대화 세션 데이터가 MongoDB에 저장되어 있을 때
    When GET /rag/sessions/:sessionId로 세션 상세를 조회하면
    Then 모든 턴의 isCancelled 필드가 false로 기본 반환되고 오류 없이 정상 응답한다.
    """
    now = datetime.now(UTC)
    legacy_mongo_doc: dict[str, Any] = {
        "sessionId": "legacy-session-ac1",
        "userId": "user-1",
        "title": "레거시 대화 세션",
        "turns": [
            {"role": "user", "content": "레거시 질문 1", "createdAt": now},
            {"role": "assistant", "content": "레거시 답변 1", "createdAt": now},
            {"role": "user", "content": "레거시 질문 2", "createdAt": now},
            {"role": "assistant", "content": "레거시 답변 2", "createdAt": now},
        ],
        "createdAt": now,
        "updatedAt": now,
    }
    legacy_session = ConversationSessionRepository._to_domain(legacy_mongo_doc)

    repo_mock = MagicMock(spec=ConversationSessionRepository)
    repo_mock.find_by_id_for_user = AsyncMock(return_value=legacy_session)
    app.dependency_overrides[get_conversation_session_repository] = lambda: repo_mock

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/rag/sessions/legacy-session-ac1?userId=user-1")

    assert response.status_code == 200
    body = response.json()
    assert body["sessionId"] == "legacy-session-ac1"
    assert len(body["turns"]) == 4
    for turn in body["turns"]:
        assert "isCancelled" in turn
        assert turn["isCancelled"] is False


async def test_get_sessions_with_keyword_returns_snippet_from_turn_content() -> None:
    now = datetime.now(UTC)
    session = ConversationSession.restore(
        RestoreProps(
            session_id="session-welfare-1",
            user_id="user-1",
            title="회사 복지 제도",
            turns=[
                TurnRecord(role="user", content="복지 제도가 어떻게 되나요?", created_at=now),
                TurnRecord(
                    role="assistant",
                    content="연차 유급휴가는 입사 1년 후 15일 부여됩니다.",
                    created_at=now,
                ),
            ],
            created_at=now,
            updated_at=now,
        )
    )
    repo_mock = MagicMock(spec=ConversationSessionRepository)
    repo_mock.find_by_user_id = AsyncMock(return_value=[session])
    app.dependency_overrides[get_conversation_session_repository] = lambda: repo_mock

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/rag/sessions?userId=user-1&keyword=유급휴가")

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["sessionId"] == "session-welfare-1"
    assert body[0]["title"] == "회사 복지 제도"
    assert "snippet" in body[0]
    assert body[0]["snippet"] is not None
    assert "유급휴가" in body[0]["snippet"]
    repo_mock.find_by_user_id.assert_awaited_once_with("user-1", 1, 10, "유급휴가")


async def test_get_sessions_without_keyword_returns_snippet_none() -> None:
    now = datetime.now(UTC)
    session = ConversationSession.restore(
        RestoreProps(
            session_id="session-general-1",
            user_id="user-1",
            title="일반 대화",
            turns=[
                TurnRecord(role="user", content="안녕", created_at=now),
                TurnRecord(role="assistant", content="안녕하세요!", created_at=now),
            ],
            created_at=now,
            updated_at=now,
        )
    )
    repo_mock = MagicMock(spec=ConversationSessionRepository)
    repo_mock.find_by_user_id = AsyncMock(return_value=[session])
    app.dependency_overrides[get_conversation_session_repository] = lambda: repo_mock

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/rag/sessions?userId=user-1")

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["sessionId"] == "session-general-1"
    assert body[0]["snippet"] is None


async def test_get_sessions_special_character_keyword_and_owner_isolation() -> None:
    now = datetime.now(UTC)
    session = ConversationSession.restore(
        RestoreProps(
            session_id="session-user1-special",
            user_id="user-1",
            title="특수문자 질의",
            turns=[
                TurnRecord(
                    role="assistant",
                    content="쿠폰 코드 [긴급] $100 적용 완료되었습니다.",
                    created_at=now,
                ),
            ],
            created_at=now,
            updated_at=now,
        )
    )
    repo_mock = MagicMock(spec=ConversationSessionRepository)
    repo_mock.find_by_user_id = AsyncMock(return_value=[session])
    app.dependency_overrides[get_conversation_session_repository] = lambda: repo_mock

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(
            "/rag/sessions?userId=user-1&keyword=[긴급] $100"
        )

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["sessionId"] == "session-user1-special"
    assert body[0]["snippet"] is not None
    assert "[긴급] $100" in body[0]["snippet"]
    repo_mock.find_by_user_id.assert_awaited_once_with("user-1", 1, 10, "[긴급] $100")


async def test_get_sessions_owner_isolation_does_not_return_other_user_sessions() -> None:
    now = datetime.now(UTC)
    session_user2 = ConversationSession.restore(
        RestoreProps(
            session_id="session-user-2-welfare",
            user_id="user-2",
            title="다른 사용자 대화",
            turns=[
                TurnRecord(
                    role="assistant",
                    content="연차 유급휴가는 입사 1년 후 15일 부여됩니다.",
                    created_at=now,
                ),
            ],
            created_at=now,
            updated_at=now,
        )
    )

    all_sessions = [session_user2]

    repo_mock = MagicMock(spec=ConversationSessionRepository)

    async def mock_find_by_user_id(
        user_id: str, page: int = 1, limit: int = 10, keyword: str | None = None
    ) -> list[ConversationSession]:
        return [s for s in all_sessions if s.get_user_id() == user_id]

    repo_mock.find_by_user_id = AsyncMock(side_effect=mock_find_by_user_id)
    app.dependency_overrides[get_conversation_session_repository] = lambda: repo_mock

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/rag/sessions?userId=user-1&keyword=유급휴가")

    assert response.status_code == 200
    body = response.json()
    assert body == []
    repo_mock.find_by_user_id.assert_awaited_once_with("user-1", 1, 10, "유급휴가")


@pytest.mark.parametrize("empty_kw", ["", "   "])
async def test_get_sessions_with_empty_or_whitespace_keyword_returns_snippet_none(
    empty_kw: str,
) -> None:
    now = datetime.now(UTC)
    session = ConversationSession.restore(
        RestoreProps(
            session_id="session-empty-kw",
            user_id="user-1",
            title="일반 대화",
            turns=[
                TurnRecord(role="user", content="안녕하세요", created_at=now),
                TurnRecord(role="assistant", content="반갑습니다.", created_at=now),
            ],
            created_at=now,
            updated_at=now,
        )
    )
    repo_mock = MagicMock(spec=ConversationSessionRepository)
    repo_mock.find_by_user_id = AsyncMock(return_value=[session])
    app.dependency_overrides[get_conversation_session_repository] = lambda: repo_mock

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(f"/rag/sessions?userId=user-1&keyword={empty_kw}")

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["snippet"] is None


@pytest.mark.parametrize("kw", ["매출 (Q3)", "[긴급] 배송", "$100"])
async def test_get_sessions_spec_special_character_keywords(kw: str) -> None:
    now = datetime.now(UTC)
    session = ConversationSession.restore(
        RestoreProps(
            session_id=f"session-spec-{kw}",
            user_id="user-1",
            title="특수문자 테스트",
            turns=[
                TurnRecord(
                    role="assistant",
                    content=f"결과 안내: 본문에 {kw} 내용이 포함되어 있습니다.",
                    created_at=now,
                ),
            ],
            created_at=now,
            updated_at=now,
        )
    )
    repo_mock = MagicMock(spec=ConversationSessionRepository)
    repo_mock.find_by_user_id = AsyncMock(return_value=[session])
    app.dependency_overrides[get_conversation_session_repository] = lambda: repo_mock

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/rag/sessions", params={"userId": "user-1", "keyword": kw})

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["snippet"] is not None
    assert kw in body[0]["snippet"]
    repo_mock.find_by_user_id.assert_awaited_once_with("user-1", 1, 10, kw)







