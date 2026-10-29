from datetime import UTC, datetime
from typing import Any

from ai_service.rag.repository import ConversationSessionRepository
from ai_service.rag.schemas import (
    ConversationSession,
    ConversationTurn,
    RestoreProps,
    SessionDetailOut,
    SessionOut,
    TurnOut,
    TurnRecord,
)


def test_conversation_turn_with_metadata():
    sources: list[dict[str, Any]] = [
        {"fileName": "doc.pdf", "chunkIndex": 0, "documentId": "doc-1", "snippet": "내용"}
    ]
    turn = ConversationTurn.of_assistant(
        "어시스턴트 답변",
        sources=sources,
        confidence=0.95,
        missing=["추가정보"],
        iterations=2,
    )
    assert turn.role == "assistant"
    assert turn.content == "어시스턴트 답변"
    assert turn.sources == sources
    assert turn.confidence == 0.95
    assert turn.missing == ["추가정보"]
    assert turn.iterations == 2

    user_turn = ConversationTurn.of_user("유저 질문")
    assert user_turn.role == "user"
    assert user_turn.sources is None
    assert user_turn.confidence is None
    assert user_turn.missing is None
    assert user_turn.iterations is None


def test_conversation_session_append_turn_with_metadata():
    session = ConversationSession.create("user-1", "질문 1")
    sources = [{"fileName": "a.pdf", "chunkIndex": 1, "documentId": "doc-1", "snippet": "요약"}]
    updated = session.append_turn(
        "질문 1",
        "답변 1",
        sources=sources,
        confidence=0.88,
        missing=["누락"],
        iterations=3,
    )
    assert len(updated.turns) == 2
    user_turn = updated.turns[0]
    assistant_turn = updated.turns[1]

    assert user_turn.role == "user"
    assert user_turn.iterations is None
    assert assistant_turn.role == "assistant"
    assert assistant_turn.sources == sources
    assert assistant_turn.confidence == 0.88
    assert assistant_turn.missing == ["누락"]
    assert assistant_turn.iterations == 3


def test_conversation_session_restore_with_and_without_metadata():
    now = datetime.now(UTC)
    sources = [{"fileName": "a.pdf", "chunkIndex": 0, "documentId": "d1", "snippet": "s"}]
    props = RestoreProps(
        session_id="session-1",
        user_id="user-1",
        title="제목",
        turns=[
            TurnRecord(role="user", content="질문", created_at=now),
            TurnRecord(
                role="assistant",
                content="답변",
                created_at=now,
                sources=sources,
                confidence=0.9,
                missing=[],
                iterations=2,
            ),
            TurnRecord(role="assistant", content="레거시 답변", created_at=now),
        ],
        created_at=now,
        updated_at=now,
    )
    session = ConversationSession.restore(props)
    assert len(session.turns) == 3
    assert session.turns[0].sources is None
    assert session.turns[0].iterations is None
    assert session.turns[1].sources == sources
    assert session.turns[1].confidence == 0.9
    assert session.turns[1].missing == []
    assert session.turns[1].iterations == 2
    assert session.turns[2].sources is None
    assert session.turns[2].confidence is None
    assert session.turns[2].missing is None
    assert session.turns[2].iterations is None


def test_repository_to_record_and_to_domain():
    now = datetime.now(UTC)
    sources = [{"fileName": "file.txt", "chunkIndex": 0, "documentId": "doc-99", "snippet": "text"}]
    props = RestoreProps(
        session_id="s-123",
        user_id="u-456",
        title="RAG 대화",
        turns=[
            TurnRecord(role="user", content="질문", created_at=now),
            TurnRecord(
                role="assistant",
                content="답변",
                created_at=now,
                sources=sources,
                confidence=0.92,
                missing=["항목1"],
                iterations=2,
            ),
            TurnRecord(
                role="assistant",
                content="단순 답변",
                created_at=now,
                sources=sources,
                confidence=0.80,
                missing=[],
            ),
        ],
        created_at=now,
        updated_at=now,
    )
    session = ConversationSession.restore(props)
    record = ConversationSessionRepository._to_record(session)

    assert record["sessionId"] == "s-123"
    assert len(record["turns"]) == 3
    assert record["turns"][0]["role"] == "user"
    assert "sources" not in record["turns"][0]
    assert "iterations" not in record["turns"][0]
    assert record["turns"][1]["sources"] == sources
    assert record["turns"][1]["confidence"] == 0.92
    assert record["turns"][1]["missing"] == ["항목1"]
    assert record["turns"][1]["iterations"] == 2
    assert "iterations" not in record["turns"][2]

    restored = ConversationSessionRepository._to_domain(record)
    assert restored.get_session_id() == "s-123"
    assert len(restored.turns) == 3
    assert restored.turns[0].iterations is None
    assert restored.turns[1].sources == sources
    assert restored.turns[1].confidence == 0.92
    assert restored.turns[1].missing == ["항목1"]
    assert restored.turns[1].iterations == 2
    assert restored.turns[2].iterations is None


def test_dto_serialization_with_metadata():
    now = datetime.now(UTC)
    sources = [{"fileName": "file.txt", "chunkIndex": 0, "documentId": "doc-99", "snippet": "text"}]
    turn = TurnOut(
        role="assistant",
        content="답변 내용",
        createdAt=now,
        sources=sources,
        confidence=0.85,
        missing=["누락정보"],
        iterations=2,
    )
    data = turn.model_dump(by_alias=True)
    assert data["role"] == "assistant"
    assert data["content"] == "답변 내용"
    assert data["createdAt"] == now
    assert data["sources"] == sources
    assert data["confidence"] == 0.85
    assert data["missing"] == ["누락정보"]
    assert data["iterations"] == 2

    session = ConversationSession.restore(
        RestoreProps(
            session_id="sess-1",
            user_id="user-1",
            title="세션",
            turns=[
                TurnRecord(role="user", content="질문", created_at=now),
                TurnRecord(
                    role="assistant",
                    content="답변",
                    created_at=now,
                    sources=sources,
                    confidence=0.85,
                    missing=["누락정보"],
                    iterations=2,
                ),
            ],
            created_at=now,
            updated_at=now,
        )
    )
    detail = SessionDetailOut.from_domain(session)
    detail_data = detail.model_dump(by_alias=True)
    assert detail_data["turns"][0]["sources"] is None
    assert detail_data["turns"][0]["iterations"] is None
    assert detail_data["turns"][1]["sources"] == sources
    assert detail_data["turns"][1]["confidence"] == 0.85
    assert detail_data["turns"][1]["missing"] == ["누락정보"]
    assert detail_data["turns"][1]["iterations"] == 2


def test_conversation_session_bookmark_lifecycle_and_legacy_restore():
    created = ConversationSession.create("user-1", "첫 질문")
    assert created.is_bookmarked is False

    bookmarked = created.with_bookmark(True)
    assert bookmarked.is_bookmarked is True
    assert bookmarked.get_session_id() == created.get_session_id()

    appended = bookmarked.append_turn("질문", "답변")
    assert appended.is_bookmarked is True

    record = ConversationSessionRepository._to_record(appended)
    assert record["isBookmarked"] is True

    restored = ConversationSessionRepository._to_domain(record)
    assert restored.is_bookmarked is True

    legacy_record = dict(record)
    legacy_record.pop("isBookmarked")
    restored_legacy = ConversationSessionRepository._to_domain(legacy_record)
    assert restored_legacy.is_bookmarked is False

    out = SessionOut.from_domain(restored).model_dump(by_alias=True)
    assert out["isBookmarked"] is True


def test_conversation_turn_cancellation():
    turn = ConversationTurn.of_assistant("중단된 답변", is_cancelled=True)
    assert turn.is_cancelled is True

    user_turn = ConversationTurn.of_user("질문")
    assert user_turn.is_cancelled is False

    default_turn = ConversationTurn.of_assistant("정상 답변")
    assert default_turn.is_cancelled is False


def test_conversation_session_append_turn_with_cancellation():
    session = ConversationSession.create("user-1", "질문 1")
    updated = session.append_turn("질문 1", "중단된 답변", is_cancelled=True)
    assert len(updated.turns) == 2
    assert updated.turns[0].is_cancelled is False
    assert updated.turns[1].is_cancelled is True


def test_conversation_session_restore_and_repository_with_is_cancelled():
    now = datetime.now(UTC)
    props = RestoreProps(
        session_id="s-cancel-1",
        user_id="u-1",
        title="중단 세션",
        turns=[
            TurnRecord(role="user", content="질문", created_at=now),
            TurnRecord(
                role="assistant",
                content="중단된 답변",
                created_at=now,
                is_cancelled=True,
            ),
            TurnRecord(
                role="assistant",
                content="정상 완료 답변",
                created_at=now,
                is_cancelled=False,
            ),
        ],
        created_at=now,
        updated_at=now,
    )
    session = ConversationSession.restore(props)
    assert session.turns[0].is_cancelled is False
    assert session.turns[1].is_cancelled is True
    assert session.turns[2].is_cancelled is False

    record = ConversationSessionRepository._to_record(session)
    assert record["turns"][0].get("isCancelled") is None
    assert record["turns"][1].get("isCancelled") is True
    assert record["turns"][2].get("isCancelled") is None

    restored = ConversationSessionRepository._to_domain(record)
    assert restored.turns[0].is_cancelled is False
    assert restored.turns[1].is_cancelled is True
    assert restored.turns[2].is_cancelled is False

    legacy_record = {
        "sessionId": "s-legacy-1",
        "userId": "u-1",
        "title": "레거시 세션",
        "turns": [
            {"role": "user", "content": "질문", "createdAt": now},
            {"role": "assistant", "content": "레거시 답변", "createdAt": now},
        ],
        "createdAt": now,
        "updatedAt": now,
    }
    restored_legacy = ConversationSessionRepository._to_domain(legacy_record)
    assert restored_legacy.turns[0].is_cancelled is False
    assert restored_legacy.turns[1].is_cancelled is False


def test_session_turn_out_serialization_with_is_cancelled():
    now = datetime.now(UTC)
    turn = ConversationTurn.of_assistant("중단된 답변", is_cancelled=True)
    turn_out = TurnOut.from_domain(turn)
    data = turn_out.model_dump(by_alias=True)
    assert data["isCancelled"] is True

    normal_turn = ConversationTurn.of_assistant("정상 답변")
    normal_turn_out = TurnOut.from_domain(normal_turn)
    normal_data = normal_turn_out.model_dump(by_alias=True)
    assert normal_data["isCancelled"] is False

    session = ConversationSession.restore(
        RestoreProps(
            session_id="sess-cancel-1",
            user_id="user-1",
            title="세션",
            turns=[
                TurnRecord(role="user", content="질문", created_at=now),
                TurnRecord(
                    role="assistant",
                    content="중단된 답변",
                    created_at=now,
                    is_cancelled=True,
                ),
            ],
            created_at=now,
            updated_at=now,
        )
    )
    detail = SessionDetailOut.from_domain(session)
    detail_data = detail.model_dump(by_alias=True)
    assert detail_data["turns"][0]["isCancelled"] is False
    assert detail_data["turns"][1]["isCancelled"] is True


def test_session_out_snippet_extraction():
    now = datetime.now(UTC)
    session = ConversationSession.restore(
        RestoreProps(
            session_id="s-snippet-1",
            user_id="u-1",
            title="회사 복지 제도",
            turns=[
                TurnRecord(role="user", content="복지 제도가 궁금합니다.", created_at=now),
                TurnRecord(
                    role="assistant",
                    content=(
                        "연차 유급휴가는 입사 1년 후\r\n"
                        "15일   부여됩니다.\n추가 문의는 인사팀으로 연락하세요."
                    ),
                    created_at=now,
                ),
            ],
            created_at=now,
            updated_at=now,
        )
    )

    # 1. 키워드 없음 -> snippet None
    out_no_kw = SessionOut.from_domain(session)
    assert out_no_kw.snippet is None

    # 2. 공백 키워드 -> snippet None
    out_empty_kw = SessionOut.from_domain(session, keyword="   ")
    assert out_empty_kw.snippet is None

    # 3. 매칭 안 되는 키워드 -> snippet None
    out_not_matched = SessionOut.from_domain(session, keyword="퇴직금")
    assert out_not_matched.snippet is None

    # 4. 매칭되는 키워드 -> 개행/공백 정리 및 주변 문맥 포함
    out_matched = SessionOut.from_domain(session, keyword="유급휴가")
    assert out_matched.snippet is not None
    assert "유급휴가" in out_matched.snippet
    assert "\n" not in out_matched.snippet
    assert "\r" not in out_matched.snippet
    assert "  " not in out_matched.snippet
    assert len(out_matched.snippet) <= 100

    # 5. 대소문자 무시 매칭
    session_case = ConversationSession.restore(
        RestoreProps(
            session_id="s-snippet-2",
            user_id="u-1",
            title="API 질문",
            turns=[
                TurnRecord(
                    role="assistant",
                    content="FastAPI RESTful Endpoint Guide",
                    created_at=now,
                )
            ],
            created_at=now,
            updated_at=now,
        )
    )
    out_case = SessionOut.from_domain(session_case, keyword="fastapi")
    assert out_case.snippet is not None
    assert "FastAPI" in out_case.snippet

    # 6. 특수문자 키워드 검색
    session_special = ConversationSession.restore(
        RestoreProps(
            session_id="s-snippet-3",
            user_id="u-1",
            title="특수문자 테스트",
            turns=[
                TurnRecord(
                    role="assistant",
                    content="가격 정책 안내: [긴급] 프로모션 상품은 $100 할인됩니다.",
                    created_at=now,
                )
            ],
            created_at=now,
            updated_at=now,
        )
    )
    out_special = SessionOut.from_domain(session_special, keyword="[긴급] 프로모션")
    assert out_special.snippet is not None
    assert "[긴급] 프로모션" in out_special.snippet

    out_dollar = SessionOut.from_domain(session_special, keyword="$100")
    assert out_dollar.snippet is not None
    assert "$100" in out_dollar.snippet




