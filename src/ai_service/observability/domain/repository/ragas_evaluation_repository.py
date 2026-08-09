from typing import Protocol

from ai_service.observability.domain.ragas_evaluation import RagasEvaluation


class IRagasEvaluationRepository(Protocol):
    async def persist(self, evaluation: RagasEvaluation) -> None: ...

    async def find_recent(self, limit: int) -> list[RagasEvaluation]: ...
