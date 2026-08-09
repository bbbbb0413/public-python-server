from datetime import UTC, datetime, timedelta

import pytest

from ai_service.llm_gateway.domain.repository.llm_cost_log_repository import LlmCostLog
from ai_service.llm_gateway.infrastructure.persistence.llm_cost_log_repository_impl import (
    LlmCostLogRepositoryImpl,
)

pytestmark = pytest.mark.integration


async def test_persist_and_sum_by_model(mongo_test_db) -> None:  # type: ignore[no-untyped-def]
    repo = LlmCostLogRepositoryImpl(mongo_test_db)
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
            prompt_tokens=100,
            completion_tokens=50,
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
            prompt_tokens=10,
            completion_tokens=10,
            cost_usd=0.5,
            fallback_used=False,
            attempted_models=["model-b"],
            created_at=now,
        )
    )

    results = await repo.sum_by_model(now - timedelta(hours=1), now + timedelta(hours=1))

    by_model = {r.model: r.total_cost_usd for r in results}
    assert by_model["model-a"] == pytest.approx(4.0)
    assert by_model["model-b"] == pytest.approx(0.5)


async def test_sum_by_model_excludes_out_of_range(mongo_test_db) -> None:  # type: ignore[no-untyped-def]
    repo = LlmCostLogRepositoryImpl(mongo_test_db)
    old = datetime(2020, 1, 1, tzinfo=UTC)

    await repo.persist(
        LlmCostLog(
            model="model-old",
            feature="qa",
            tenant="default",
            prompt_tokens=1,
            completion_tokens=1,
            cost_usd=99.0,
            fallback_used=False,
            attempted_models=["model-old"],
            created_at=old,
        )
    )

    now = datetime.now(UTC)
    results = await repo.sum_by_model(now - timedelta(hours=1), now + timedelta(hours=1))

    assert all(r.model != "model-old" for r in results)
