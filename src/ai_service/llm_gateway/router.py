from datetime import UTC, datetime, timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, Query

from ai_service.core.config import Settings, get_settings
from ai_service.llm_gateway.dependencies import (
    CircuitBreakerDep,
    LlmCostLogRepositoryDep,
)
from ai_service.llm_gateway.schemas import BreakerStatusOut, CostSummaryOut

router = APIRouter(prefix="/llm-gateway", tags=["llm-gateway"])


@router.get("/costs", response_model=CostSummaryOut, response_model_by_alias=True)
async def get_costs(
    repo: LlmCostLogRepositoryDep,
    from_: str | None = Query(default=None, alias="from"),
    to: str | None = Query(default=None),
) -> CostSummaryOut:
    to_date = datetime.fromisoformat(to) if to else datetime.now(UTC)
    from_date = datetime.fromisoformat(from_) if from_ else to_date - timedelta(days=7)

    items = await repo.sum_by_model(from_date, to_date)
    return CostSummaryOut.of(items, from_date, to_date)


@router.get("/breakers", response_model=list[BreakerStatusOut], response_model_by_alias=True)
async def get_breaker_statuses(
    breaker: CircuitBreakerDep, settings: Annotated[Settings, Depends(get_settings)]
) -> list[BreakerStatusOut]:
    raw = settings.llm_fallback_chain or ""
    models = [m.strip() for m in raw.split(",") if m.strip()]

    statuses: list[BreakerStatusOut] = []
    for model in models:
        snapshot = await breaker.get_state(model)
        statuses.append(BreakerStatusOut.from_snapshot(snapshot))
    return statuses


__all__ = ["router"]
