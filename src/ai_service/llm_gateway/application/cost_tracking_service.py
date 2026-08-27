import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from ai_service.llm_gateway.repository import LlmCostLog
from ai_service.llm_gateway.schemas import TokenUsage

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelCostEntry:
    prompt: float
    completion: float


@dataclass(frozen=True)
class TrackParams:
    model: str
    feature: str
    tenant: str
    usage: TokenUsage
    fallback_used: bool
    attempted_models: list[str]


def parse_cost_table(raw: str | None) -> dict[str, ModelCostEntry]:
    if not raw:
        return {}
    parsed: dict[str, dict[str, float]] = json.loads(raw)
    return {model: ModelCostEntry(**entry) for model, entry in parsed.items()}


class CostTrackingService:
    def __init__(self, repo: Any, cost_table: dict[str, ModelCostEntry]) -> None:
        self._repo = repo
        self._cost_table = cost_table

    async def track(self, params: TrackParams) -> None:
        cost_usd = self._calc_cost(params.model, params.usage)

        try:
            await self._repo.persist(
                LlmCostLog(
                    model=params.model,
                    feature=params.feature,
                    tenant=params.tenant,
                    prompt_tokens=params.usage.prompt_tokens,
                    completion_tokens=params.usage.completion_tokens,
                    cost_usd=cost_usd,
                    fallback_used=params.fallback_used,
                    attempted_models=params.attempted_models,
                    created_at=datetime.now(UTC),
                )
            )
        except Exception as e:  # noqa: BLE001 - 비용 로그 실패는 요청 흐름을 막지 않음
            logger.error("비용 로그 저장 실패: %s", e)

    def _calc_cost(self, model: str, usage: TokenUsage) -> float:
        entry = self._cost_table.get(model)
        if entry is None:
            return 0.0
        return (
            usage.prompt_tokens * entry.prompt + usage.completion_tokens * entry.completion
        ) / 1_000_000
