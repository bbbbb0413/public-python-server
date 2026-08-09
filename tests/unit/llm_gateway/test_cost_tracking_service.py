from ai_service.llm_gateway.application.cost_tracking_service import (
    CostTrackingService,
    ModelCostEntry,
    TrackParams,
    parse_cost_table,
)
from ai_service.llm_gateway.domain.vo.token_usage import TokenUsage
from tests.unit.llm_gateway.fakes import FailingLlmCostLogRepository, FakeLlmCostLogRepository


def test_parse_cost_table_from_json() -> None:
    raw = '{"model-a": {"prompt": 1.5, "completion": 2.0}}'

    table = parse_cost_table(raw)

    assert table == {"model-a": ModelCostEntry(prompt=1.5, completion=2.0)}


def test_parse_cost_table_none_returns_empty() -> None:
    assert parse_cost_table(None) == {}


async def test_track_persists_calculated_cost() -> None:
    repo = FakeLlmCostLogRepository()
    table = {"model-a": ModelCostEntry(prompt=1.0, completion=2.0)}
    service = CostTrackingService(repo, table)

    await service.track(
        TrackParams(
            model="model-a",
            feature="qa",
            tenant="default",
            usage=TokenUsage.of(1000, 500),
            fallback_used=False,
            attempted_models=["model-a"],
        )
    )

    assert len(repo.logs) == 1
    log = repo.logs[0]
    assert log.cost_usd == (1000 * 1.0 + 500 * 2.0) / 1_000_000
    assert log.model == "model-a"
    assert log.fallback_used is False


async def test_track_unknown_model_costs_zero() -> None:
    repo = FakeLlmCostLogRepository()
    service = CostTrackingService(repo, {})

    await service.track(
        TrackParams(
            model="unknown-model",
            feature="qa",
            tenant="default",
            usage=TokenUsage.of(100, 100),
            fallback_used=False,
            attempted_models=["unknown-model"],
        )
    )

    assert repo.logs[0].cost_usd == 0.0


async def test_track_swallows_repository_errors() -> None:
    service = CostTrackingService(FailingLlmCostLogRepository(), {})

    await service.track(
        TrackParams(
            model="model-a",
            feature="qa",
            tenant="default",
            usage=TokenUsage.of(1, 1),
            fallback_used=False,
            attempted_models=["model-a"],
        )
    )
