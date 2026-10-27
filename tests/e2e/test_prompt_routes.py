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


async def test_list_prompts_isolation_by_user_id(shared_repo) -> None:  # type: ignore[no-untyped-def]
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Create global v1
        await client.post(
            "/prompts",
            json={"name": "rag-qa-system", "content": "global-v1", "variables": ["context"]},
        )
        # Create user-A v1
        await client.post(
            "/prompts",
            json={
                "name": "rag-qa-system",
                "content": "user-a-v1",
                "variables": ["context"],
                "userId": "user-A",
            },
        )
        # Create user-B v1
        await client.post(
            "/prompts",
            json={
                "name": "rag-qa-system",
                "content": "user-b-v1",
                "variables": ["context"],
                "userId": "user-B",
            },
        )

        # 1. user-A requests: should get only user-A v1 (total 1), not global v1 and not user-B v1
        res_a = await client.get("/prompts/rag-qa-system", params={"userId": "user-A"})
        assert res_a.status_code == 200
        items_a = res_a.json()
        assert len(items_a) == 1
        assert [item["version"] for item in items_a] == [1]
        assert items_a[0]["userId"] == "user-A"

        # 2. No userId requests (admin global): should get only global v1
        res_global = await client.get("/prompts/rag-qa-system")
        assert res_global.status_code == 200
        items_global = res_global.json()
        assert len(items_global) == 1
        assert items_global[0]["version"] == 1
        assert items_global[0].get("userId") is None

        # 3. Non-existent name query: should return []
        res_none = await client.get("/prompts/non-existent", params={"userId": "user-A"})
        assert res_none.status_code == 200
        assert res_none.json() == []


async def test_list_prompts_when_only_user_prompts_exist(shared_repo) -> None:  # type: ignore[no-untyped-def]
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Create user-A v1
        await client.post(
            "/prompts",
            json={
                "name": "rag-qa-system",
                "content": "user-a-v1",
                "variables": ["context"],
                "userId": "user-A",
            },
        )
        # Create user-B v2
        await client.post(
            "/prompts",
            json={
                "name": "rag-qa-system",
                "content": "user-b-v2",
                "variables": ["context"],
                "userId": "user-B",
            },
        )

        res = await client.get("/prompts/rag-qa-system", params={"userId": "user-A"})
        assert res.status_code == 200
        items = res.json()
        assert len(items) == 1
        assert items[0]["version"] == 1
        assert items[0]["userId"] == "user-A"


async def test_activate_same_version_number_across_users_is_isolated(shared_repo) -> None:  # type: ignore[no-untyped-def]
    from ai_service.prompt.schemas import PromptTemplate

    await shared_repo.persist(
        PromptTemplate.create(name="rag-qa-system", content="global-v1", version=1, user_id=None)
    )
    await shared_repo.persist(
        PromptTemplate.create(
            name="rag-qa-system", content="user-a-v1", version=1, user_id="user-A"
        )
    )
    await shared_repo.persist(
        PromptTemplate.create(
            name="rag-qa-system", content="user-b-v1", version=1, user_id="user-B"
        )
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.patch(
            "/prompts/rag-qa-system/1/activate", params={"userId": "user-B"}
        )
        assert res.status_code == 200
        body = res.json()
        assert body["version"] == 1
        assert body["isActive"] is True
        assert body["userId"] == "user-B"
        assert body["content"] == "user-b-v1"


async def test_delete_prompt_version_and_fallback_flow(shared_repo) -> None:  # type: ignore[no-untyped-def]
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post(
            "/prompts",
            json={
                "name": "rag-qa-system",
                "content": "user-a-v1",
                "variables": ["context"],
                "userId": "user-A",
            },
        )
        await client.post(
            "/prompts",
            json={
                "name": "rag-qa-system",
                "content": "user-a-v2",
                "variables": ["context"],
                "userId": "user-A",
            },
        )
        activate_res = await client.patch(
            "/prompts/rag-qa-system/2/activate", params={"userId": "user-A"}
        )
        assert activate_res.status_code == 200

        del_res = await client.delete(
            "/prompts/rag-qa-system/2", params={"userId": "user-A"}
        )
        assert del_res.status_code == 204

        list_res = await client.get(
            "/prompts/rag-qa-system", params={"userId": "user-A"}
        )
        assert [item["version"] for item in list_res.json()] == [1]

        active_res = await client.get(
            "/prompts/rag-qa-system/active", params={"userId": "user-A"}
        )
        assert active_res.status_code == 200
        assert active_res.json()["version"] == 0


async def test_delete_prompt_version_falls_back_to_global_active_prompt(shared_repo) -> None:  # type: ignore[no-untyped-def]
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post(
            "/prompts",
            json={
                "name": "rag-qa-system",
                "content": "global-v1",
                "variables": ["context"],
            },
        )
        await client.patch("/prompts/rag-qa-system/1/activate")

        user_res = await client.post(
            "/prompts",
            json={
                "name": "rag-qa-system",
                "content": "user-a-v1",
                "variables": ["context"],
                "userId": "user-A",
            },
        )
        user_ver = user_res.json()["version"]
        await client.patch(
            f"/prompts/rag-qa-system/{user_ver}/activate", params={"userId": "user-A"}
        )

        del_res = await client.delete(
            f"/prompts/rag-qa-system/{user_ver}", params={"userId": "user-A"}
        )
        assert del_res.status_code == 204

        active_res = await client.get(
            "/prompts/rag-qa-system/active", params={"userId": "user-A"}
        )
        assert active_res.status_code == 200
        assert active_res.json()["version"] == 1
        assert active_res.json()["content"] == "global-v1"


async def test_delete_prompt_version_returns_404_when_not_found_or_other_user(shared_repo) -> None:  # type: ignore[no-untyped-def]
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post(
            "/prompts",
            json={
                "name": "rag-qa-system",
                "content": "user-a-v1",
                "variables": ["context"],
                "userId": "user-A",
            },
        )

        other_user_res = await client.delete(
            "/prompts/rag-qa-system/1", params={"userId": "user-B"}
        )
        assert other_user_res.status_code == 404

        global_scope_res = await client.delete("/prompts/rag-qa-system/1")
        assert global_scope_res.status_code == 404

        missing_ver_res = await client.delete(
            "/prompts/rag-qa-system/999", params={"userId": "user-A"}
        )
        assert missing_ver_res.status_code == 404

        list_res = await client.get(
            "/prompts/rag-qa-system", params={"userId": "user-A"}
        )
        assert len(list_res.json()) == 1
        assert list_res.json()[0]["version"] == 1


async def test_delete_active_route_does_not_conflict_with_delete_version(shared_repo) -> None:  # type: ignore[no-untyped-def]
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post(
            "/prompts",
            json={
                "name": "rag-qa-system",
                "content": "user-a-v1",
                "variables": ["context"],
                "userId": "user-A",
            },
        )
        await client.patch(
            "/prompts/rag-qa-system/1/activate", params={"userId": "user-A"}
        )

        reset_res = await client.delete(
            "/prompts/rag-qa-system/active", params={"userId": "user-A"}
        )
        assert reset_res.status_code == 204
        list_res = await client.get(
            "/prompts/rag-qa-system", params={"userId": "user-A"}
        )
        assert len(list_res.json()) == 1
        assert list_res.json()[0]["isActive"] is False


async def test_delete_active_route_without_user_id_deactivates_global_active(shared_repo) -> None:  # type: ignore[no-untyped-def]
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post(
            "/prompts",
            json={
                "name": "rag-qa-system",
                "content": "global-v1",
                "variables": ["context"],
            },
        )
        await client.patch("/prompts/rag-qa-system/1/activate")

        active_res = await client.get("/prompts/rag-qa-system/active")
        assert active_res.status_code == 200
        assert active_res.json()["version"] == 1

        reset_res = await client.delete("/prompts/rag-qa-system/active")
        assert reset_res.status_code == 204

        active_after = await client.get("/prompts/rag-qa-system/active")
        assert active_after.status_code == 200
        assert active_after.json()["version"] == 0

