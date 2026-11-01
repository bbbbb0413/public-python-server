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
