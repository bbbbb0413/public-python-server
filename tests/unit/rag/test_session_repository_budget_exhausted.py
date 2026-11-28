from datetime import UTC, datetime

from ai_service.rag.repository import ConversationSessionRepository
from ai_service.rag.schemas import ConversationSession


def test_repository_to_record_with_budget_exhausted():
    session = ConversationSession.create("user-1", "예산 소진 저장 테스트")
    session = session.append_turn(
        user_content="질문",
        assistant_content="예산 소진 답변",
        is_budget_exhausted=True,
    )

    record = ConversationSessionRepository._to_record(session)
    assert len(record["turns"]) == 2
    user_turn = record["turns"][0]
    assistant_turn = record["turns"][1]

    assert "isBudgetExhausted" not in user_turn
    assert assistant_turn["isBudgetExhausted"] is True


def test_repository_to_record_without_budget_exhausted():
    session = ConversationSession.create("user-1", "정상 저장 테스트")
    session = session.append_turn(
        user_content="질문",
        assistant_content="정상 답변",
        is_budget_exhausted=False,
    )

    record = ConversationSessionRepository._to_record(session)
    assistant_turn = record["turns"][1]
    assert "isBudgetExhausted" not in assistant_turn


def test_repository_to_domain_with_budget_exhausted():
    now = datetime.now(UTC)
    record = {
        "sessionId": "session-1",
        "userId": "user-1",
        "title": "세션 복원",
        "turns": [
            {"role": "user", "content": "질문", "createdAt": now},
            {
                "role": "assistant",
                "content": "예산 소진 답변",
                "createdAt": now,
                "isBudgetExhausted": True,
            },
        ],
        "createdAt": now,
        "updatedAt": now,
    }

    session = ConversationSessionRepository._to_domain(record)
    assert session.turns[1].is_budget_exhausted is True


def test_repository_to_domain_legacy_fallback_when_field_missing():
    now = datetime.now(UTC)
    record = {
        "sessionId": "session-legacy",
        "userId": "user-1",
        "title": "레거시 세션 복원",
        "turns": [
            {"role": "user", "content": "질문", "createdAt": now},
            {
                "role": "assistant",
                "content": "과거 답변 (필드 없음)",
                "createdAt": now,
            },
        ],
        "createdAt": now,
        "updatedAt": now,
    }

    session = ConversationSessionRepository._to_domain(record)
    assert session.turns[1].is_budget_exhausted is False
