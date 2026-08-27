from datetime import UTC, datetime

import pytest
from httpx import ASGITransport, AsyncClient

from ai_service.main import app
from ai_service.observability.dependencies import get_ragas_evaluation_repository
from ai_service.observability.schemas import RagasEvaluation
from tests.unit.observability.fakes import FakeRagasEvaluationRepository


@pytest.fixture(autouse=True)
def _clear_overrides():
    yield
    app.dependency_overrides.clear()


async def test_get_evals_returns_camel_case_payload() -> None:
    repo = FakeRagasEvaluationRepository()
    repo.saved.append(
        RagasEvaluation(
            trace_id="t1",
            question="q",
            faithfulness=0.5,
            answer_relevancy=0.6,
            context_precision=0.5,
            sampled_at=datetime(2026, 1, 1, tzinfo=UTC),
        )
    )
    app.dependency_overrides[get_ragas_evaluation_repository] = lambda: repo

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/observability/ragas-evals")

    assert response.status_code == 200
    body = response.json()
    assert body["data"][0]["traceId"] == "t1"
    assert body["data"][0]["answerRelevancy"] == 0.6
