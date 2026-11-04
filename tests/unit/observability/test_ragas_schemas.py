from datetime import UTC, datetime

from ai_service.observability.schemas import (
    RagasEvaluation,
    RagasEvaluationOut,
)


def test_ragas_evaluation_domain_defaults() -> None:
    eval_item = RagasEvaluation(
        trace_id="t1",
        question="q1",
        faithfulness=0.8,
        answer_relevancy=0.9,
        context_precision=0.7,
        sampled_at=datetime.now(UTC),
    )
    assert eval_item.answer == ""
    assert eval_item.contexts == []


def test_ragas_evaluation_domain_with_answer_and_contexts() -> None:
    eval_item = RagasEvaluation(
        trace_id="t1",
        question="q1",
        faithfulness=0.8,
        answer_relevancy=0.9,
        context_precision=0.7,
        sampled_at=datetime.now(UTC),
        answer="답변 본문",
        contexts=["컨텍스트 1", "컨텍스트 2"],
    )
    assert eval_item.answer == "답변 본문"
    assert eval_item.contexts == ["컨텍스트 1", "컨텍스트 2"]


def test_ragas_evaluation_out_from_domain() -> None:
    eval_item = RagasEvaluation(
        trace_id="t1",
        question="q1",
        faithfulness=0.8,
        answer_relevancy=0.9,
        context_precision=0.7,
        sampled_at=datetime(2026, 1, 1, tzinfo=UTC),
        answer="답변 본문",
        contexts=["컨텍스트 1", "컨텍스트 2"],
    )
    dto = RagasEvaluationOut.from_domain(eval_item)
    assert dto.answer == "답변 본문"
    assert dto.contexts == ["컨텍스트 1", "컨텍스트 2"]

    dumped = dto.model_dump(by_alias=True)
    assert dumped["answer"] == "답변 본문"
    assert dumped["contexts"] == ["컨텍스트 1", "컨텍스트 2"]
