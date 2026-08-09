from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from ai_service.shared_kernel.value_object import ValueObject

TurnRole = Literal["user", "assistant"]


@dataclass(frozen=True)
class TurnValue:
    role: TurnRole
    content: str
    created_at: datetime


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
    def of_assistant(cls, content: str) -> "ConversationTurn":
        return cls(TurnValue(role="assistant", content=content, created_at=datetime.now(UTC)))

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
