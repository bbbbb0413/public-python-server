from datetime import UTC, datetime

from ai_service.rag.repository import ConversationSessionRepository
from ai_service.rag.schemas import ConversationSession


def test_repository_to_record_serializes_refined_queries() -> None:
    session = ConversationSession.create("user-1", "첫 질문")
    updated = session.append_turn(
        user_content="질문",
        assistant_content="답변",
        refined_queries=["정제 쿼리 1", "정제 쿼리 2"],
    )
    record = ConversationSessionRepository._to_record(updated)
    assistant_record = record["turns"][1]
    assert assistant_record["refinedQueries"] == ["정제 쿼리 1", "정제 쿼리 2"]


def test_repository_to_record_omits_refined_queries_when_none() -> None:
    session = ConversationSession.create("user-1", "첫 질문")
    updated = session.append_turn(
        user_content="질문",
        assistant_content="답변",
        refined_queries=None,
    )
    record = ConversationSessionRepository._to_record(updated)
    assistant_record = record["turns"][1]
    assert "refinedQueries" not in assistant_record


def test_repository_to_domain_deserializes_refined_queries() -> None:
    now = datetime.now(UTC)
    record = {
        "sessionId": "session-1",
        "userId": "user-1",
        "title": "질문",
        "turns": [
            {
                "role": "user",
                "content": "질문",
                "createdAt": now,
            },
            {
                "role": "assistant",
                "content": "답변",
                "createdAt": now,
                "refinedQueries": ["복원 정제 쿼리"],
            },
        ],
        "createdAt": now,
        "updatedAt": now,
        "isBookmarked": False,
    }
    session = ConversationSessionRepository._to_domain(record)
    assert session.turns[1].refined_queries == ["복원 정제 쿼리"]


def test_repository_to_domain_backward_compatibility_without_refined_queries() -> None:
    now = datetime.now(UTC)
    record = {
        "sessionId": "session-1",
        "userId": "user-1",
        "title": "질문",
        "turns": [
            {
                "role": "assistant",
                "content": "답변",
                "createdAt": now,
            },
        ],
        "createdAt": now,
        "updatedAt": now,
        "isBookmarked": False,
    }
    session = ConversationSessionRepository._to_domain(record)
    assert session.turns[0].refined_queries is None


def test_repository_roundtrip_with_empty_refined_queries() -> None:
    session = ConversationSession.create("user-1", "1회차 만족 질문")
    updated = session.append_turn(
        user_content="질문",
        assistant_content="1회차에 해결된 답변",
        refined_queries=[],
    )
    record = ConversationSessionRepository._to_record(updated)
    assistant_record = record["turns"][1]
    assert assistant_record["refinedQueries"] == []

    restored = ConversationSessionRepository._to_domain(record)
    assert restored.turns[1].refined_queries == []

