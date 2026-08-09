from datetime import UTC, datetime

import pytest

from ai_service.observability.domain.ragas_evaluation import RagasEvaluation
from ai_service.observability.infrastructure.persistence.ragas_evaluation_repository_impl import (
    RagasEvaluationRepositoryImpl,
)

pytestmark = pytest.mark.integration


async def test_find_recent_returns_newest_first(mongo_test_db) -> None:  # type: ignore[no-untyped-def]
    repo = RagasEvaluationRepositoryImpl(mongo_test_db)
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
    repo = RagasEvaluationRepositoryImpl(mongo_test_db)
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
