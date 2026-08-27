from typing import Any

from motor.motor_asyncio import AsyncIOMotorDatabase

from ai_service.observability.schemas import RagasEvaluation

COLLECTION_NAME = "ragas_evaluations"


class RagasEvaluationRepository:
    def __init__(self, db: AsyncIOMotorDatabase[dict[str, Any]]) -> None:
        self._collection = db[COLLECTION_NAME]

    async def persist(self, evaluation: RagasEvaluation) -> None:
        await self._collection.insert_one(
            {
                "traceId": evaluation.trace_id,
                "question": evaluation.question,
                "faithfulness": evaluation.faithfulness,
                "answerRelevancy": evaluation.answer_relevancy,
                "contextPrecision": evaluation.context_precision,
                "sampledAt": evaluation.sampled_at,
            }
        )

    async def find_recent(self, limit: int) -> list[RagasEvaluation]:
        cursor = self._collection.find({}, {"_id": 0}).sort("sampledAt", -1).limit(limit)
        records = await cursor.to_list(length=None)
        return [
            RagasEvaluation(
                trace_id=r["traceId"],
                question=r["question"],
                faithfulness=r["faithfulness"],
                answer_relevancy=r["answerRelevancy"],
                context_precision=r["contextPrecision"],
                sampled_at=r["sampledAt"],
            )
            for r in records
        ]


# Backward compatibility alias
RagasEvaluationRepositoryImpl = RagasEvaluationRepository

__all__ = ["COLLECTION_NAME", "RagasEvaluationRepository", "RagasEvaluationRepositoryImpl"]
