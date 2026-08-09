from dataclasses import dataclass
from datetime import datetime
from typing import Protocol


@dataclass(frozen=True)
class LlmCostLog:
    model: str
    feature: str
    tenant: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    fallback_used: bool
    attempted_models: list[str]
    created_at: datetime


@dataclass(frozen=True)
class ModelCostSum:
    model: str
    total_cost_usd: float


class ILlmCostLogRepository(Protocol):
    async def persist(self, log: LlmCostLog) -> None: ...

    async def sum_by_model(self, from_: datetime, to: datetime) -> list[ModelCostSum]: ...
