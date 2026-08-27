from datetime import UTC, datetime, timedelta

import pytest

from ai_service.llm_gateway.repository import LlmCostLog, LlmCostLogRepository

pytestmark = pytest.mark.integration


async def test_persist_and_sum_by_model(mongo_test_db) -> None:  # type: ignore[no-untyped-def]
    repo = LlmCostLogRepository(mongo_test_db)
    now = datetime.now(UTC)

    await repo.persist(
        LlmCostLog(
            model="model-a",
            feature="qa",
            tenant="default",
            prompt_tokens=100,
            completion_tokens=50,
            cost_usd=1.5,
            fallback_used=False,
            attempted_models=["model-a"],
            created_at=now,
        )
    )
    await repo.persist(
        LlmCostLog(
            model="model-a",
            feature="qa",
            tenant="default",
            prompt_tokens=200,
            completion_tokens=100,
            cost_usd=2.5,
            fallback_used=False,
            attempted_models=["model-a"],
            created_at=now,
        )
    )
    await repo.persist(
        LlmCostLog(
            model="model-b",
            feature="qa",
            tenant="default",
            prompt_tokens=50,
            completion_tokens=20,
            cost_usd=0.5,
            fallback_used=False,
            attempted_models=["model-b"],
            created_at=now,
        )
    )

    from_ = now - timedelta(hours=1)
    to = now + timedelta(hours=1)
    sums = await repo.sum_by_model(from_, to)

    assert len(sums) == 2
    model_a_sum = next(s for s in sums if s.model == "model-a")
    model_b_sum = next(s for s in sums if s.model == "model-b")
    assert model_a_sum.total_cost_usd == 4.0
    assert model_b_sum.total_cost_usd == 0.5


async def test_sum_by_model_filters_by_date(mongo_test_db) -> None:  # type: ignore[no-untyped-def]
    repo = LlmCostLogRepository(mongo_test_db)
    old = datetime(2025, 1, 1, tzinfo=UTC)
    recent = datetime(2026, 1, 1, tzinfo=UTC)

    await repo.persist(
        LlmCostLog(
            model="model-a",
            feature="qa",
            tenant="default",
            prompt_tokens=10,
            completion_tokens=10,
            cost_usd=1.0,
            fallback_used=False,
            attempted_models=["model-a"],
            created_at=old,
        )
    )
    await repo.persist(
        LlmCostLog(
            model="model-a",
            feature="qa",
            tenant="default",
            prompt_tokens=10,
            completion_tokens=10,
            cost_usd=2.0,
            fallback_used=False,
            attempted_models=["model-a"],
            created_at=recent,
        )
    )

    from_ = datetime(2026, 1, 1, tzinfo=UTC)
    to = datetime(2026, 1, 2, tzinfo=UTC)
    sums = await repo.sum_by_model(from_, to)

    assert len(sums) == 1
    assert sums[0].total_cost_usd == 2.0
