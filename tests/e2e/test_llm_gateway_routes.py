from datetime import datetime

import pytest
from httpx import ASGITransport, AsyncClient

from ai_service.core.config import Settings, get_settings
from ai_service.llm_gateway.dependencies import (
    get_circuit_breaker,
    get_llm_cost_log_repository,
)
from ai_service.llm_gateway.schemas import ModelCostSum
from ai_service.main import app
from tests.unit.llm_gateway.fakes import FakeCircuitBreaker, FakeLlmCostLogRepository


@pytest.fixture(autouse=True)
def _clear_overrides():
    yield
    app.dependency_overrides.clear()


async def test_get_costs_returns_camel_case_payload() -> None:
    repo = FakeLlmCostLogRepository()

    async def fake_sum_by_model(from_, to):  # noqa: ANN001, ANN202
        return [
            ModelCostSum(
                model="model-a",
                total_cost_usd=1.23,
                prompt_tokens=100,
                completion_tokens=50,
                call_count=2,
            )
        ]

    repo.sum_by_model = fake_sum_by_model  # type: ignore[method-assign]
    app.dependency_overrides[get_llm_cost_log_repository] = lambda: repo

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/llm-gateway/costs")

    assert response.status_code == 200
    body = response.json()
    assert body["items"] == [
        {
            "model": "model-a",
            "totalCostUsd": 1.23,
            "promptTokens": 100,
            "completionTokens": 50,
            "totalTokens": 150,
            "callCount": 2,
        }
    ]
    assert "from" in body
    assert "to" in body


async def test_get_costs_passes_from_and_to_query_params() -> None:
    repo = FakeLlmCostLogRepository()
    captured_dates: dict[str, datetime] = {}

    async def fake_sum_by_model(from_date, to_date):  # noqa: ANN001, ANN202
        captured_dates["from"] = from_date
        captured_dates["to"] = to_date
        return []

    repo.sum_by_model = fake_sum_by_model  # type: ignore[method-assign]
    app.dependency_overrides[get_llm_cost_log_repository] = lambda: repo

    transport = ASGITransport(app=app)
    from_param = "2026-11-01T00:00:00+00:00"
    to_param = "2026-11-02T00:00:00+00:00"
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(
            "/llm-gateway/costs",
            params={"from": from_param, "to": to_param},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["items"] == []
    assert captured_dates["from"] == datetime.fromisoformat(from_param)
    assert captured_dates["to"] == datetime.fromisoformat(to_param)


async def test_get_costs_handles_space_decoded_timezone_offset() -> None:
    repo = FakeLlmCostLogRepository()
    captured_dates: dict[str, datetime] = {}

    async def fake_sum_by_model(from_date, to_date):  # noqa: ANN001, ANN202
        captured_dates["from"] = from_date
        captured_dates["to"] = to_date
        return []

    repo.sum_by_model = fake_sum_by_model  # type: ignore[method-assign]
    app.dependency_overrides[get_llm_cost_log_repository] = lambda: repo

    transport = ASGITransport(app=app)
    # When "+" is decoded into space by proxies/gateways
    from_raw = "2026-11-01T00:00:00 09:00"
    to_raw = "2026-11-02T00:00:00 09:00"
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(
            "/llm-gateway/costs",
            params={"from": from_raw, "to": to_raw},
        )

    assert response.status_code == 200
    assert captured_dates["from"] == datetime.fromisoformat("2026-11-01T00:00:00+09:00")
    assert captured_dates["to"] == datetime.fromisoformat("2026-11-02T00:00:00+09:00")


async def test_get_costs_default_period_when_only_to_provided() -> None:
    repo = FakeLlmCostLogRepository()
    captured_dates: dict[str, datetime] = {}

    async def fake_sum_by_model(from_date, to_date):  # noqa: ANN001, ANN202
        captured_dates["from"] = from_date
        captured_dates["to"] = to_date
        return []

    repo.sum_by_model = fake_sum_by_model  # type: ignore[method-assign]
    app.dependency_overrides[get_llm_cost_log_repository] = lambda: repo

    transport = ASGITransport(app=app)
    to_param = "2026-11-08T12:00:00+00:00"
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/llm-gateway/costs", params={"to": to_param})

    assert response.status_code == 200
    expected_to = datetime.fromisoformat(to_param)
    assert captured_dates["to"] == expected_to
    assert (expected_to - captured_dates["from"]).total_seconds() == 7 * 86400


async def test_get_costs_multiple_models_serializes_all_metrics() -> None:
    repo = FakeLlmCostLogRepository()

    async def fake_sum_by_model(from_, to):  # noqa: ANN001, ANN202
        return [
            ModelCostSum(
                model="gpt-4o",
                total_cost_usd=2.5,
                prompt_tokens=2000,
                completion_tokens=500,
                call_count=10,
            ),
            ModelCostSum(
                model="claude-3-5-sonnet",
                total_cost_usd=1.1,
                prompt_tokens=800,
                completion_tokens=200,
                call_count=4,
            ),
        ]

    repo.sum_by_model = fake_sum_by_model  # type: ignore[method-assign]
    app.dependency_overrides[get_llm_cost_log_repository] = lambda: repo

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/llm-gateway/costs")

    assert response.status_code == 200
    body = response.json()
    assert len(body["items"]) == 2
    assert body["items"][0] == {
        "model": "gpt-4o",
        "totalCostUsd": 2.5,
        "promptTokens": 2000,
        "completionTokens": 500,
        "totalTokens": 2500,
        "callCount": 10,
    }
    assert body["items"][1] == {
        "model": "claude-3-5-sonnet",
        "totalCostUsd": 1.1,
        "promptTokens": 800,
        "completionTokens": 200,
        "totalTokens": 1000,
        "callCount": 4,
    }



async def test_get_breakers_returns_status_for_configured_chain() -> None:
    breaker = FakeCircuitBreaker()
    app.dependency_overrides[get_circuit_breaker] = lambda: breaker
    app.dependency_overrides[get_settings] = lambda: Settings(llm_fallback_chain="model-a,model-b")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/llm-gateway/breakers")

    assert response.status_code == 200
    body = response.json()
    assert [item["model"] for item in body] == ["model-a", "model-b"]
    assert all(item["status"] == "closed" for item in body)
    assert all("failureCount" in item and "openedAt" in item for item in body)
