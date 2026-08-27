from dataclasses import dataclass
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


@dataclass(frozen=True)
class RagasEvalPayload:
    trace_id: str
    question: str
    answer: str
    contexts: list[str]


@dataclass(frozen=True)
class RagasEvaluation:
    trace_id: str
    question: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    sampled_at: datetime


@dataclass(frozen=True)
class RagasScores:
    faithfulness: float
    answer_relevancy: float
    context_precision: float


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


__all__ = [
    "RagasEvalPayload",
    "RagasEvaluation",
    "RagasEvaluationListOut",
    "RagasEvaluationOut",
    "RagasScores",
]
