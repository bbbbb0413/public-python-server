from ai_service.observability.domain.ragas_evaluation import RagasEvaluation


class FakeRagasEvaluationRepository:
    def __init__(self) -> None:
        self.saved: list[RagasEvaluation] = []

    async def persist(self, evaluation: RagasEvaluation) -> None:
        self.saved.append(evaluation)

    async def find_recent(self, limit: int) -> list[RagasEvaluation]:
        return self.saved[:limit]
