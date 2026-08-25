from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

from ai_service.rag.domain.vo.conversation_turn import ConversationTurn, TurnValue
from ai_service.rag.domain.vo.session_id import SessionId
from ai_service.shared_kernel.aggregate_root import AggregateRoot

MAX_TITLE_LENGTH = 50


@dataclass(frozen=True)
class TurnRecord:
    role: Literal["user", "assistant"]
    content: str
    created_at: datetime
    sources: list[dict[str, Any]] | None = None
    confidence: float | None = None
    missing: list[str] | None = None


@dataclass(frozen=True)
class RestoreProps:
    session_id: str
    user_id: str
    title: str
    turns: list[TurnRecord]
    created_at: datetime
    updated_at: datetime


class ConversationSession(AggregateRoot):
    def __init__(
        self,
        session_id: SessionId,
        user_id: str,
        title: str,
        turns: list[ConversationTurn],
        created_at: datetime,
        updated_at: datetime,
    ) -> None:
        super().__init__()
        self._session_id = session_id
        self._user_id = user_id
        self._title = title
        self._turns = turns
        self._created_at = created_at
        self._updated_at = updated_at

    @classmethod
    def create(cls, user_id: str, first_question: str) -> "ConversationSession":
        now = datetime.now(UTC)
        title = first_question[:MAX_TITLE_LENGTH]
        return cls(SessionId.generate(), user_id, title, [], now, now)

    @classmethod
    def restore(cls, props: RestoreProps) -> "ConversationSession":
        turns = [
            ConversationTurn.restore(
                TurnValue(
                    role=t.role,
                    content=t.content,
                    created_at=t.created_at,
                    sources=t.sources,
                    confidence=t.confidence,
                    missing=t.missing,
                )
            )
            for t in props.turns
        ]
        return cls(
            SessionId.of(props.session_id),
            props.user_id,
            props.title,
            turns,
            props.created_at,
            props.updated_at,
        )

    def append_turn(
        self,
        user_content: str,
        assistant_content: str,
        sources: list[dict[str, Any]] | None = None,
        confidence: float | None = None,
        missing: list[str] | None = None,
    ) -> "ConversationSession":
        new_turns = [
            *self._turns,
            ConversationTurn.of_user(user_content),
            ConversationTurn.of_assistant(
                assistant_content,
                sources=sources,
                confidence=confidence,
                missing=missing,
            ),
        ]
        return ConversationSession(
            self._session_id,
            self._user_id,
            self._title,
            new_turns,
            self._created_at,
            datetime.now(UTC),
        )

    def get_history(self) -> list[dict[str, str]]:
        return [{"role": t.role, "content": t.content} for t in self._turns]

    def get_session_id(self) -> str:
        return self._session_id.get_value()

    def get_user_id(self) -> str:
        return self._user_id

    @property
    def title(self) -> str:
        return self._title

    @property
    def turns(self) -> list[ConversationTurn]:
        return list(self._turns)

    @property
    def created_at(self) -> datetime:
        return self._created_at

    @property
    def updated_at(self) -> datetime:
        return self._updated_at
