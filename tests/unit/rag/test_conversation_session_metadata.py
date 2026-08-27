from datetime import UTC, datetime
from typing import Any

from ai_service.rag.repository import ConversationSessionRepository
from ai_service.rag.schemas import (
    ConversationSession,
    ConversationTurn,
    RestoreProps,
    SessionDetailOut,
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
    )
    assert turn.role == "assistant"
    assert turn.content == "어시스턴트 답변"
    assert turn.sources == sources
    assert turn.confidence == 0.95
    assert turn.missing == ["추가정보"]

    user_turn = ConversationTurn.of_user("유저 질문")
    assert user_turn.role == "user"
    assert user_turn.sources is None
    assert user_turn.confidence is None
    assert user_turn.missing is None


def test_conversation_session_append_turn_with_metadata():
    session = ConversationSession.create("user-1", "질문 1")
    sources = [{"fileName": "a.pdf", "chunkIndex": 1, "documentId": "doc-1", "snippet": "요약"}]
    updated = session.append_turn(
        "질문 1",
        "답변 1",
        sources=sources,
        confidence=0.88,
        missing=["누락"],
    )
    assert len(updated.turns) == 2
    user_turn = updated.turns[0]
    assistant_turn = updated.turns[1]

    assert user_turn.role == "user"
    assert assistant_turn.role == "assistant"
    assert assistant_turn.sources == sources
    assert assistant_turn.confidence == 0.88
    assert assistant_turn.missing == ["누락"]


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
            ),
            TurnRecord(role="assistant", content="레거시 답변", created_at=now),
        ],
        created_at=now,
        updated_at=now,
    )
    session = ConversationSession.restore(props)
    assert len(session.turns) == 3
    assert session.turns[0].sources is None
    assert session.turns[1].sources == sources
    assert session.turns[1].confidence == 0.9
    assert session.turns[1].missing == []
    assert session.turns[2].sources is None
    assert session.turns[2].confidence is None
    assert session.turns[2].missing is None


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
            ),
        ],
        created_at=now,
        updated_at=now,
    )
    session = ConversationSession.restore(props)
    record = ConversationSessionRepository._to_record(session)

    assert record["sessionId"] == "s-123"
    assert len(record["turns"]) == 2
    assert record["turns"][0]["role"] == "user"
    assert "sources" not in record["turns"][0]
    assert record["turns"][1]["sources"] == sources
    assert record["turns"][1]["confidence"] == 0.92
    assert record["turns"][1]["missing"] == ["항목1"]

    restored = ConversationSessionRepository._to_domain(record)
    assert restored.get_session_id() == "s-123"
    assert len(restored.turns) == 2
    assert restored.turns[1].sources == sources
    assert restored.turns[1].confidence == 0.92
    assert restored.turns[1].missing == ["항목1"]


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
    )
    data = turn.model_dump(by_alias=True)
    assert data["role"] == "assistant"
    assert data["content"] == "답변 내용"
    assert data["createdAt"] == now
    assert data["sources"] == sources
    assert data["confidence"] == 0.85
    assert data["missing"] == ["누락정보"]

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
                ),
            ],
            created_at=now,
            updated_at=now,
        )
    )
    detail = SessionDetailOut.from_domain(session)
    detail_data = detail.model_dump(by_alias=True)
    assert detail_data["turns"][0]["sources"] is None
    assert detail_data["turns"][1]["sources"] == sources
    assert detail_data["turns"][1]["confidence"] == 0.85
    assert detail_data["turns"][1]["missing"] == ["누락정보"]
