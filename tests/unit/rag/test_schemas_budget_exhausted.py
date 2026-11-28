from datetime import UTC, datetime

from ai_service.rag.schemas import (
    ConversationSession,
    ConversationTurn,
    RestoreProps,
    SessionTurnOut,
    TurnRecord,
    TurnValue,
)


def test_turn_value_and_record_default_is_budget_exhausted():
    now = datetime.now(UTC)
    turn_val = TurnValue(role="assistant", content="답변", created_at=now)
    assert turn_val.is_budget_exhausted is False

    turn_rec = TurnRecord(role="assistant", content="답변", created_at=now)
    assert turn_rec.is_budget_exhausted is False


def test_conversation_turn_of_assistant_with_is_budget_exhausted():
    turn = ConversationTurn.of_assistant(
        content="예산 소진 답변",
        is_budget_exhausted=True,
    )
    assert turn.is_budget_exhausted is True

    turn_normal = ConversationTurn.of_assistant(
        content="정상 답변",
    )
    assert turn_normal.is_budget_exhausted is False


def test_conversation_session_append_turn_with_is_budget_exhausted():
    session = ConversationSession.create("user-1", "질문입니다")
    updated = session.append_turn(
        user_content="질문입니다",
        assistant_content="예산 소진 답변",
        is_budget_exhausted=True,
    )
    assert len(updated.turns) == 2
    assistant_turn = updated.turns[1]
    assert assistant_turn.role == "assistant"
    assert assistant_turn.is_budget_exhausted is True


def test_conversation_session_restore_with_is_budget_exhausted():
    now = datetime.now(UTC)
    props = RestoreProps(
        session_id="session-1",
        user_id="user-1",
        title="세션 복원 테스트",
        turns=[
            TurnRecord(role="user", content="질문", created_at=now),
            TurnRecord(
                role="assistant",
                content="복원된 답변",
                created_at=now,
                is_budget_exhausted=True,
            ),
        ],
        created_at=now,
        updated_at=now,
    )
    session = ConversationSession.restore(props)
    assert session.turns[1].is_budget_exhausted is True


def test_session_turn_out_from_domain_with_is_budget_exhausted():
    turn = ConversationTurn.of_assistant(
        content="예산 소진 답변",
        is_budget_exhausted=True,
    )
    dto = SessionTurnOut.from_domain(turn)
    assert dto.is_budget_exhausted is True
    assert dto.model_dump(by_alias=True)["isBudgetExhausted"] is True

    # 기본값 테스트
    turn_default = ConversationTurn.of_assistant(content="일반 답변")
    dto_default = SessionTurnOut.from_domain(turn_default)
    assert dto_default.is_budget_exhausted is False
    assert dto_default.model_dump(by_alias=True)["isBudgetExhausted"] is False
