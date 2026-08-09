from dataclasses import dataclass
from typing import Literal

LlmRole = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class LlmMessage:
    role: LlmRole
    content: str


@dataclass(frozen=True)
class LlmOptions:
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
