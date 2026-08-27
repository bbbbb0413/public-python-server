from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

BreakerStatus = Literal["closed", "open", "half-open"]


@dataclass(frozen=True)
class CircuitBreakerSnapshot:
    model: str
    status: BreakerStatus
    failure_count: int
    opened_at: int | None


@dataclass(frozen=True)
class LlmMessage:
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(frozen=True)
class LlmOptions:
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    stream: bool = True


@dataclass(frozen=True)
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    @classmethod
    def of(cls, prompt_tokens: int, completion_tokens: int) -> "TokenUsage":
        if prompt_tokens < 0:
            raise ValueError("promptTokens는 0 이상이어야 합니다.")
        if completion_tokens < 0:
            raise ValueError("completionTokens는 0 이상이어야 합니다.")
        return cls(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )

    def total(self) -> int:
        return self.total_tokens


@dataclass(frozen=True)
class ModelRoute:
    model: str
    temperature: float = 0.7
    max_tokens: int = 1024

    @classmethod
    def of(cls, model: str, temperature: float = 0.7, max_tokens: int = 1024) -> "ModelRoute":
        if not model or not model.strip():
            raise ValueError("모델명은 빈 값일 수 없습니다.")
        return cls(model=model, temperature=temperature, max_tokens=max_tokens)

    def get_value(self) -> str:
        return self.model


@dataclass(frozen=True)
class ModelCostSum:
    model: str
    total_cost_usd: float


@dataclass(frozen=True)
class GatewayCallCommand:
    messages: list[LlmMessage]
    feature: str = "qa"
    tenant: str = "default"
    stream: bool = True
    session_id: str | None = None
    user_id: str | None = None
    prompt_name: str | None = None
    preferred_model: str | None = None


class CostSummaryItem(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    model: str
    total_cost_usd: float = Field(alias="totalCostUsd")


class CostSummaryOut(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    items: list[CostSummaryItem]
    from_: str = Field(alias="from")
    to: str

    @classmethod
    def of(
        cls,
        items: list[ModelCostSum],
        from_date: datetime,
        to_date: datetime,
    ) -> "CostSummaryOut":
        return cls(
            items=[CostSummaryItem(model=i.model, totalCostUsd=i.total_cost_usd) for i in items],
            **{"from": from_date.isoformat(), "to": to_date.isoformat()},
        )


class BreakerStatusOut(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    model: str
    status: str
    failure_count: int = Field(alias="failureCount")
    opened_at: int | None = Field(alias="openedAt")

    @classmethod
    def from_snapshot(cls, snapshot: CircuitBreakerSnapshot) -> "BreakerStatusOut":
        return cls(
            model=snapshot.model,
            status=snapshot.status,
            failureCount=snapshot.failure_count,
            openedAt=snapshot.opened_at,
        )


__all__ = [
    "BreakerStatus",
    "BreakerStatusOut",
    "CircuitBreakerSnapshot",
    "CostSummaryItem",
    "CostSummaryOut",
    "GatewayCallCommand",
    "LlmMessage",
    "LlmOptions",
    "ModelCostSum",
    "ModelRoute",
    "TokenUsage",
]
