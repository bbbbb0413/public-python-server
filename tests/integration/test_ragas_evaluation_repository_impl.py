from datetime import UTC, datetime

import pytest

from ai_service.observability.repository import RagasEvaluationRepository
from ai_service.observability.schemas import RagasEvaluation

pytestmark = pytest.mark.integration


async def test_find_recent_returns_newest_first(mongo_test_db) -> None:  # type: ignore[no-untyped-def]
    repo = RagasEvaluationRepository(mongo_test_db)
    older = RagasEvaluation(
        trace_id="t1",
        question="q1",
        faithfulness=0.5,
        answer_relevancy=0.5,
        context_precision=0.5,
        sampled_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    newer = RagasEvaluation(
        trace_id="t2",
        question="q2",
        faithfulness=0.9,
        answer_relevancy=0.9,
        context_precision=0.9,
        sampled_at=datetime(2026, 1, 2, tzinfo=UTC),
    )
    await repo.persist(older)
    await repo.persist(newer)

    results = await repo.find_recent(10)

    assert [r.trace_id for r in results] == ["t2", "t1"]


async def test_find_recent_respects_limit(mongo_test_db) -> None:  # type: ignore[no-untyped-def]
    repo = RagasEvaluationRepository(mongo_test_db)
    for i in range(5):
        await repo.persist(
            RagasEvaluation(
                trace_id=f"t{i}",
                question="q",
                faithfulness=0.5,
                answer_relevancy=0.5,
                context_precision=0.5,
                sampled_at=datetime(2026, 1, 1 + i, tzinfo=UTC),
            )
        )

    results = await repo.find_recent(2)

    assert len(results) == 2


async def test_find_recent_restores_answer_and_contexts_and_fallbacks(mongo_test_db) -> None:  # type: ignore[no-untyped-def]
    repo = RagasEvaluationRepository(mongo_test_db)
    # 1. 새 형식 엔티티 저장
    item = RagasEvaluation(
        trace_id="new-t",
        question="신규 질문",
        faithfulness=0.8,
        answer_relevancy=0.8,
        context_precision=0.8,
        sampled_at=datetime(2026, 1, 2, tzinfo=UTC),
        answer="신규 답변",
        contexts=["근거 1"],
    )
    await repo.persist(item)

    # 2. 레거시 형식 (answer, contexts 키가 없는 문서) 직접 삽입
    await mongo_test_db["ragas_evaluations"].insert_one(
        {
            "traceId": "legacy-t",
            "question": "과거 질문",
            "faithfulness": 0.5,
            "answerRelevancy": 0.5,
            "contextPrecision": 0.5,
            "sampledAt": datetime(2026, 1, 1, tzinfo=UTC),
        }
    )

    results = await repo.find_recent(10)
    assert len(results) == 2
    assert results[0].trace_id == "new-t"
    assert results[0].answer == "신규 답변"
    assert results[0].contexts == ["근거 1"]
    assert results[1].trace_id == "legacy-t"
    assert results[1].answer == ""
    assert results[1].contexts == []
