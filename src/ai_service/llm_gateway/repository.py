from dataclasses import dataclass
from datetime import datetime
from typing import Any

from motor.motor_asyncio import AsyncIOMotorDatabase

from ai_service.llm_gateway.schemas import ModelCostSum

COLLECTION_NAME = "llm_cost_logs"


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


class LlmCostLogRepository:
    def __init__(self, db: AsyncIOMotorDatabase[dict[str, Any]]) -> None:
        self._collection = db[COLLECTION_NAME]

    async def persist(self, log: LlmCostLog) -> None:
        await self._collection.insert_one(
            {
                "model": log.model,
                "feature": log.feature,
                "tenant": log.tenant,
                "promptTokens": log.prompt_tokens,
                "completionTokens": log.completion_tokens,
                "costUsd": log.cost_usd,
                "fallbackUsed": log.fallback_used,
                "attemptedModels": log.attempted_models,
                "createdAt": log.created_at,
            }
        )

    async def sum_by_model(self, from_: datetime, to: datetime) -> list[ModelCostSum]:
        cursor = self._collection.aggregate(
            [
                {"$match": {"createdAt": {"$gte": from_, "$lte": to}}},
                {"$group": {"_id": "$model", "totalCostUsd": {"$sum": "$costUsd"}}},
                {"$sort": {"totalCostUsd": -1}},
            ]
        )
        results = await cursor.to_list(length=None)
        return [ModelCostSum(model=r["_id"], total_cost_usd=r["totalCostUsd"]) for r in results]


# Backward compatibility alias
LlmCostLogRepositoryImpl = LlmCostLogRepository

__all__ = ["COLLECTION_NAME", "LlmCostLog", "LlmCostLogRepository", "LlmCostLogRepositoryImpl"]
