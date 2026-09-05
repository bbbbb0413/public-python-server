import pytest
from httpx import ASGITransport, AsyncClient

from ai_service.main import app
from ai_service.prompt.dependencies import get_prompt_template_repository
from tests.unit.prompt.fakes import FakePromptTemplateRepository


@pytest.fixture
def shared_repo():  # noqa: ANN201
    repo = FakePromptTemplateRepository()
    app.dependency_overrides[get_prompt_template_repository] = lambda: repo
    yield repo
    app.dependency_overrides.clear()


async def test_create_list_and_activate_flow(shared_repo) -> None:  # type: ignore[no-untyped-def]
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        create_response = await client.post(
            "/prompts",
            json={
                "name": "rag-qa-system",
                "content": "hello {{context}}",
                "variables": ["context"],
            },
        )
        assert create_response.status_code == 200
        created = create_response.json()
        assert created["version"] == 1
        assert created["isActive"] is False

        list_response = await client.get("/prompts/rag-qa-system")
        assert list_response.status_code == 200
        assert len(list_response.json()) == 1

        activate_response = await client.patch("/prompts/rag-qa-system/1/activate")
        assert activate_response.status_code == 200
        assert activate_response.json()["isActive"] is True

        active_response = await client.get("/prompts/rag-qa-system/active")
        assert active_response.status_code == 200
        assert active_response.json()["version"] == 1


async def test_activate_missing_version_returns_404(shared_repo) -> None:  # type: ignore[no-untyped-def]
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.patch("/prompts/rag-qa-system/99/activate")

    assert response.status_code == 404


async def test_get_active_falls_back_to_default_prompt(shared_repo) -> None:  # type: ignore[no-untyped-def]
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/prompts/never-created/active")

    assert response.status_code == 200
    assert response.json()["version"] == 0


async def test_create_rejects_invalid_name(shared_repo) -> None:  # type: ignore[no-untyped-def]
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/prompts", json={"name": "Invalid Name!", "content": "c", "variables": []}
        )

    assert response.status_code == 422


async def test_list_versions_user_isolation(shared_repo) -> None:  # type: ignore[no-untyped-def]
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post(
            "/prompts",
            json={"name": "rag-qa-system", "content": "global-v1", "variables": ["context"]},
        )
        await client.post(
            "/prompts",
            json={
                "name": "rag-qa-system",
                "content": "userA-v2",
                "variables": ["context"],
                "userId": "user-A",
            },
        )
        await client.post(
            "/prompts",
            json={
                "name": "rag-qa-system",
                "content": "userB-v3",
                "variables": ["context"],
                "userId": "user-B",
            },
        )

        # 1. user-A should see global-v1 + userA-v2, but NOT userB-v3
        resp_a = await client.get("/prompts/rag-qa-system", params={"userId": "user-A"})
        assert resp_a.status_code == 200
        data_a = resp_a.json()
        versions_a = [item["version"] for item in data_a]
        assert set(versions_a) == {1, 2}

        # 2. No userId query param should return only global-v1
        resp_anon = await client.get("/prompts/rag-qa-system")
        assert resp_anon.status_code == 200
        data_anon = resp_anon.json()
        versions_anon = [item["version"] for item in data_anon]
        assert versions_anon == [1]
