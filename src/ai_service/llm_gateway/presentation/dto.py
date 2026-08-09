from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from ai_service.llm_gateway.domain.model.circuit_breaker_state import CircuitBreakerSnapshot
from ai_service.llm_gateway.domain.repository.llm_cost_log_repository import ModelCostSum


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
