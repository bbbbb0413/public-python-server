from datetime import UTC, datetime

from ai_service.rag.schemas import (
    ConversationSession,
    ConversationTurn,
    RestoreProps,
    SessionTurnOut,
    TurnRecord,
    TurnValue,
)


def test_turn_value_refined_queries_default_none() -> None:
    tv = TurnValue(
        role="assistant",
        content="답변",
        created_at=datetime.now(UTC),
    )
    assert tv.refined_queries is None


def test_conversation_turn_of_assistant_with_refined_queries() -> None:
    turn = ConversationTurn.of_assistant(
        content="답변",
        sources=[],
        confidence=0.9,
        missing=[],
        iterations=2,
        refined_queries=["보완 검색 쿼리 1", "보완 검색 쿼리 2"],
    )
    assert turn.refined_queries == ["보완 검색 쿼리 1", "보완 검색 쿼리 2"]


def test_conversation_turn_restore_refined_queries() -> None:
    tv = TurnValue(
        role="assistant",
        content="답변",
        created_at=datetime.now(UTC),
        refined_queries=["정제 쿼리"],
    )
    turn = ConversationTurn.restore(tv)
    assert turn.refined_queries == ["정제 쿼리"]


def test_conversation_session_append_turn_with_refined_queries() -> None:
    session = ConversationSession.create("user-1", "첫 질문")
    updated = session.append_turn(
        user_content="질문",
        assistant_content="답변",
        refined_queries=["보완 검색어"],
    )
    assert len(updated.turns) == 2
    assistant_turn = updated.turns[1]
    assert assistant_turn.refined_queries == ["보완 검색어"]


def test_conversation_session_restore_with_refined_queries() -> None:
    props = RestoreProps(
        session_id="session-1",
        user_id="user-1",
        title="제목",
        turns=[
            TurnRecord(
                role="user",
                content="질문",
                created_at=datetime.now(UTC),
            ),
            TurnRecord(
                role="assistant",
                content="답변",
                created_at=datetime.now(UTC),
                refined_queries=["복원된 정제 쿼리"],
            ),
        ],
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    session = ConversationSession.restore(props)
    assert len(session.turns) == 2
    assert session.turns[1].refined_queries == ["복원된 정제 쿼리"]


def test_session_turn_out_from_domain_with_refined_queries() -> None:
    turn = ConversationTurn.of_assistant(
        content="답변",
        refined_queries=["보완 질문"],
    )
    turn_out = SessionTurnOut.from_domain(turn)
    assert turn_out.refined_queries == ["보완 질문"]
    dumped = turn_out.model_dump(by_alias=True)
    assert dumped["refinedQueries"] == ["보완 질문"]


def test_session_turn_out_from_domain_with_empty_refined_queries() -> None:
    turn = ConversationTurn.of_assistant(
        content="1회차 답변",
        refined_queries=[],
    )
    turn_out = SessionTurnOut.from_domain(turn)
    assert turn_out.refined_queries == []
    dumped = turn_out.model_dump(by_alias=True)
    assert dumped["refinedQueries"] == []

