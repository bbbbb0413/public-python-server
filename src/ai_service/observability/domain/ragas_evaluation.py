from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class RagasEvaluation:
    trace_id: str
    question: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    sampled_at: datetime
