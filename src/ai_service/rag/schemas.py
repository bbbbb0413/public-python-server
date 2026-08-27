import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

MAX_TITLE_LENGTH = 50


@dataclass(frozen=True)
class SessionId:
    value: str

    @classmethod
    def generate(cls) -> "SessionId":
        return cls(str(uuid.uuid4()))

    @classmethod
    def of(cls, value: str) -> "SessionId":
        if not value or not value.strip():
            raise ValueError("세션 ID는 비어있을 수 없습니다.")
        return cls(value)

    def get_value(self) -> str:
        return self.value


@dataclass(frozen=True)
class TurnValue:
    role: Literal["user", "assistant"]
    content: str
    created_at: datetime
    sources: list[dict[str, Any]] | None = None
    confidence: float | None = None
    missing: list[str] | None = None


class ConversationTurn:
    def __init__(self, value: TurnValue) -> None:
        self._value = value

    @classmethod
    def of_user(cls, content: str) -> "ConversationTurn":
        return cls(TurnValue(role="user", content=content, created_at=datetime.now(UTC)))

    @classmethod
    def of_assistant(
        cls,
        content: str,
        sources: list[dict[str, Any]] | None = None,
        confidence: float | None = None,
        missing: list[str] | None = None,
    ) -> "ConversationTurn":
        return cls(
            TurnValue(
                role="assistant",
                content=content,
                created_at=datetime.now(UTC),
                sources=sources,
                confidence=confidence,
                missing=missing,
            )
        )

    @classmethod
    def restore(cls, value: TurnValue) -> "ConversationTurn":
        return cls(value)

    @property
    def role(self) -> Literal["user", "assistant"]:
        return self._value.role

    @property
    def content(self) -> str:
        return self._value.content

    @property
    def created_at(self) -> datetime:
        return self._value.created_at

    @property
    def sources(self) -> list[dict[str, Any]] | None:
        return self._value.sources

    @property
    def confidence(self) -> float | None:
        return self._value.confidence

    @property
    def missing(self) -> list[str] | None:
        return self._value.missing


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


class ConversationSession:
    def __init__(
        self,
        session_id: SessionId,
        user_id: str,
        title: str,
        turns: list[ConversationTurn],
        created_at: datetime,
        updated_at: datetime,
    ) -> None:
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


@dataclass(frozen=True)
class CritiqueProps:
    answered: bool
    missing: list[str] = field(default_factory=list)
    next_query: str = ""
    confidence: float = 0.0


class Critique:
    def __init__(self, value: CritiqueProps) -> None:
        if value.confidence < 0 or value.confidence > 1:
            raise ValueError("confidence는 0 이상 1 이하여야 합니다.")
        self._value = value

    @classmethod
    def of(
        cls, answered: bool, missing: list[str], next_query: str, confidence: float
    ) -> "Critique":
        return cls(CritiqueProps(answered, missing, next_query, confidence))

    def is_satisfied(self, threshold: float) -> bool:
        return self._value.answered and self._value.confidence >= threshold

    def get_next_query(self) -> str:
        return self._value.next_query

    def get_confidence(self) -> float:
        return self._value.confidence

    def get_missing(self) -> list[str]:
        return list(self._value.missing)


@dataclass(frozen=True)
class IterationBudgetProps:
    max_iterations: int
    token_budget: int
    timeout_ms: int


class IterationBudget:
    def __init__(self, value: IterationBudgetProps) -> None:
        if value.max_iterations < 1 or value.max_iterations > 10:
            raise ValueError("max_iterations는 1 이상 10 이하여야 합니다.")
        if value.token_budget <= 0:
            raise ValueError("token_budget은 양수여야 합니다.")
        if value.timeout_ms <= 0:
            raise ValueError("timeout_ms는 양수여야 합니다.")
        self._value = value

    @classmethod
    def of(cls, max_iterations: int, token_budget: int, timeout_ms: int) -> "IterationBudget":
        return cls(IterationBudgetProps(max_iterations, token_budget, timeout_ms))

    def is_exhausted(self, iterations_completed: int, tokens_used: int, elapsed_ms: float) -> bool:
        return (
            iterations_completed >= self._value.max_iterations
            or tokens_used >= self._value.token_budget
            or elapsed_ms >= self._value.timeout_ms
        )

    def get_max_iterations(self) -> int:
        return self._value.max_iterations

    def get_token_budget(self) -> int:
        return self._value.token_budget

    def get_timeout_ms(self) -> int:
        return self._value.timeout_ms


@dataclass(frozen=True)
class VerdictValue:
    allowed: bool
    reason: str
    matched_pattern: str | None = None


class GuardrailVerdict:
    def __init__(self, value: VerdictValue) -> None:
        if not value.allowed and not value.reason:
            raise ValueError("차단 판정에는 사유가 필요합니다.")
        self._value = value

    @classmethod
    def allow(cls) -> "GuardrailVerdict":
        return cls(VerdictValue(allowed=True, reason="ok"))

    @classmethod
    def block(cls, reason: str, pattern: str | None = None) -> "GuardrailVerdict":
        return cls(VerdictValue(allowed=False, reason=reason, matched_pattern=pattern))

    def is_allowed(self) -> bool:
        return self._value.allowed

    def get_reason(self) -> str:
        return self._value.reason


class SimilarityThreshold:
    def __init__(self, value: float) -> None:
        if value < 0 or value > 1:
            raise ValueError("유사도 임계값은 0과 1 사이여야 합니다.")
        self._value = value

    @classmethod
    def of(cls, value: float) -> "SimilarityThreshold":
        return cls(value)

    def get_value(self) -> float:
        return self._value


class SourceRefOut(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    file_name: str = Field(alias="fileName")
    chunk_index: int = Field(alias="chunkIndex")
    document_id: str = Field(alias="documentId")
    snippet: str | None = None


class SessionTurnOut(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    role: Literal["user", "assistant"]
    content: str
    created_at: datetime = Field(alias="createdAt")
    sources: list[dict[str, Any]] | None = None
    confidence: float | None = None
    missing: list[str] | None = None

    @classmethod
    def from_domain(cls, turn: ConversationTurn) -> "SessionTurnOut":
        return cls(
            role=turn.role,
            content=turn.content,
            createdAt=turn.created_at,
            sources=turn.sources,
            confidence=turn.confidence,
            missing=turn.missing,
        )


class SessionOut(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    session_id: str = Field(alias="sessionId")
    title: str
    updated_at: datetime = Field(alias="updatedAt")

    @classmethod
    def from_domain(cls, session: ConversationSession) -> "SessionOut":
        return cls(
            sessionId=session.get_session_id(),
            title=session.title,
            updatedAt=session.updated_at,
        )


class SessionDetailOut(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    session_id: str = Field(alias="sessionId")
    title: str
    turns: list[SessionTurnOut]
    created_at: datetime = Field(alias="createdAt")
    updated_at: datetime = Field(alias="updatedAt")

    @classmethod
    def from_domain(cls, session: ConversationSession) -> "SessionDetailOut":
        return cls(
            sessionId=session.get_session_id(),
            title=session.title,
            turns=[SessionTurnOut.from_domain(t) for t in session.turns],
            createdAt=session.created_at,
            updatedAt=session.updated_at,
        )


TurnOut = SessionTurnOut

__all__ = [
    "ConversationSession",
    "ConversationTurn",
    "Critique",
    "CritiqueProps",
    "GuardrailVerdict",
    "IterationBudget",
    "IterationBudgetProps",
    "MAX_TITLE_LENGTH",
    "RestoreProps",
    "SessionDetailOut",
    "SessionId",
    "SessionOut",
    "SessionTurnOut",
    "SimilarityThreshold",
    "SourceRefOut",
    "TurnOut",
    "TurnRecord",
    "TurnValue",
    "VerdictValue",
]
