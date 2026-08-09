from dataclasses import dataclass


@dataclass(frozen=True)
class RagasEvalPayload:
    trace_id: str
    question: str
    answer: str
    contexts: list[str]
