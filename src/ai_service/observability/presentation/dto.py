from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from ai_service.observability.domain.ragas_evaluation import RagasEvaluation


class RagasEvaluationOut(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    trace_id: str = Field(alias="traceId")
    question: str
    faithfulness: float
    answer_relevancy: float = Field(alias="answerRelevancy")
    context_precision: float = Field(alias="contextPrecision")
    sampled_at: datetime = Field(alias="sampledAt")

    @classmethod
    def from_domain(cls, evaluation: RagasEvaluation) -> "RagasEvaluationOut":
        return cls(
            traceId=evaluation.trace_id,
            question=evaluation.question,
            faithfulness=evaluation.faithfulness,
            answerRelevancy=evaluation.answer_relevancy,
            contextPrecision=evaluation.context_precision,
            sampledAt=evaluation.sampled_at,
        )


class RagasEvaluationListOut(BaseModel):
    data: list[RagasEvaluationOut]
