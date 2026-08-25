from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

from ai_service.shared_kernel.value_object import ValueObject

TurnRole = Literal["user", "assistant"]


@dataclass(frozen=True)
class TurnValue:
    role: TurnRole
    content: str
    created_at: datetime
    sources: list[dict[str, Any]] | None = None
    confidence: float | None = None
    missing: list[str] | None = None


class ConversationTurn(ValueObject[TurnValue]):
    def _validate(self, value: TurnValue) -> None:
        if not value.content or not value.content.strip():
            raise ValueError("turn content는 비어있을 수 없습니다.")
        if value.role not in ("user", "assistant"):
            raise ValueError("role은 user 또는 assistant이어야 합니다.")

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
    def role(self) -> TurnRole:
        return self.get_value().role

    @property
    def content(self) -> str:
        return self.get_value().content

    @property
    def created_at(self) -> datetime:
        return self.get_value().created_at

    @property
    def sources(self) -> list[dict[str, Any]] | None:
        return self.get_value().sources

    @property
    def confidence(self) -> float | None:
        return self.get_value().confidence

    @property
    def missing(self) -> list[str] | None:
        return self.get_value().missing
