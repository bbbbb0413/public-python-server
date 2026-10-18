import pytest

from ai_service.prompt.schemas import (
    CreatePromptIn,
    PromptTemplate,
    PromptTemplateNotFoundError,
)
from ai_service.prompt.service import RAG_QA_DEFAULT_PROMPT, PromptService
from tests.unit.prompt.fakes import FakePromptTemplateRepository


async def test_first_version_starts_at_one() -> None:
    repo = FakePromptTemplateRepository()
    service = PromptService(repo)

    template = await service.create_prompt(
        CreatePromptIn(name="rag-qa-system", content="c", variables=["context"])
    )

    assert template.version == 1


async def test_next_version_increments_from_existing_max() -> None:
    repo = FakePromptTemplateRepository()
    service = PromptService(repo)
    await service.create_prompt(CreatePromptIn(name="rag-qa-system", content="v1"))
    await service.create_prompt(CreatePromptIn(name="rag-qa-system", content="v2"))

    template = await service.create_prompt(CreatePromptIn(name="rag-qa-system", content="v3"))

    assert template.version == 3


async def test_activating_missing_version_raises() -> None:
    repo = FakePromptTemplateRepository()
    service = PromptService(repo)

    with pytest.raises(PromptTemplateNotFoundError):
        await service.activate_prompt(name="rag-qa-system", version=1)


async def test_activating_deactivates_previous_active_version() -> None:
    repo = FakePromptTemplateRepository()
    service = PromptService(repo)
    await service.create_prompt(CreatePromptIn(name="rag-qa-system", content="v1"))
    await service.create_prompt(CreatePromptIn(name="rag-qa-system", content="v2"))

    await service.activate_prompt(name="rag-qa-system", version=1)
    activated_v2 = await service.activate_prompt(name="rag-qa-system", version=2)

    v1 = await repo.find_by_name_and_version("rag-qa-system", 1)
    assert v1 is not None
    assert v1.is_active is False
    assert activated_v2.is_active is True


async def test_user_scoped_activate_does_not_touch_global_active_version() -> None:
    repo = FakePromptTemplateRepository()
    service = PromptService(repo)
    await service.create_prompt(CreatePromptIn(name="rag-qa-system", content="global-v1"))
    await service.create_prompt(
        CreatePromptIn(name="rag-qa-system", content="user-v1", userId="user-1")
    )

    global_active = await service.activate_prompt(name="rag-qa-system", version=1)
    user_active = await service.activate_prompt(
        name="rag-qa-system", version=2, user_id="user-1"
    )

    reloaded_global = await repo.find_by_name_and_version("rag-qa-system", 1)
    assert reloaded_global is not None
    assert reloaded_global.is_active is True
    assert global_active.is_active is True
    assert user_active.is_active is True


async def test_cannot_activate_another_users_version() -> None:
    repo = FakePromptTemplateRepository()
    service = PromptService(repo)
    await service.create_prompt(
        CreatePromptIn(name="rag-qa-system", content="user-v1", userId="user-1")
    )

    with pytest.raises(PromptTemplateNotFoundError):
        await service.activate_prompt(name="rag-qa-system", version=1, user_id="user-2")
    with pytest.raises(PromptTemplateNotFoundError):
        await service.activate_prompt(name="rag-qa-system", version=1)


async def test_returns_hardcoded_default_when_nothing_stored() -> None:
    repo = FakePromptTemplateRepository()
    service = PromptService(repo)

    template = await service.get_active_prompt("rag-qa-system")

    assert template.content == RAG_QA_DEFAULT_PROMPT
    assert template.version == 0
    assert template.is_active is False


async def test_prefers_global_active_over_default() -> None:
    repo = FakePromptTemplateRepository()
    active = PromptTemplate.create(name="rag-qa-system", content="global").activate()
    repo.storage[("rag-qa-system", active.version, None)] = active

    service = PromptService(repo)
    template = await service.get_active_prompt("rag-qa-system")

    assert template.content == "global"


async def test_prefers_user_specific_active_over_global() -> None:
    repo = FakePromptTemplateRepository()
    global_active = PromptTemplate.create(name="rag-qa-system", content="global").activate()
    repo.storage[("rag-qa-system", global_active.version, None)] = global_active

    user_active = PromptTemplate.create(
        name="rag-qa-system", content="for-user", version=2, user_id="user-1"
    ).activate()
    repo.storage[("rag-qa-system", user_active.version, "user-1")] = user_active

    service = PromptService(repo)
    template = await service.get_active_prompt("rag-qa-system", user_id="user-1")

    assert template.content == "for-user"


async def test_other_users_active_prompt_never_leaks_as_fallback() -> None:
    repo = FakePromptTemplateRepository()
    user1_active = PromptTemplate.create(
        name="rag-qa-system", content="user-1-only", user_id="user-1"
    ).activate()
    repo.storage[("rag-qa-system", user1_active.version, "user-1")] = user1_active

    service = PromptService(repo)
    template = await service.get_active_prompt("rag-qa-system", user_id="user-2")

    assert template.content != "user-1-only"
    assert template.content == RAG_QA_DEFAULT_PROMPT


async def test_list_versions_with_user_id_returns_global_and_own_prompts_only() -> None:
    repo = FakePromptTemplateRepository()
    service = PromptService(repo)
    await service.create_prompt(CreatePromptIn(name="rag-qa-system", content="global-v1"))
    await service.create_prompt(
        CreatePromptIn(name="rag-qa-system", content="user-A-v2", userId="user-A")
    )
    await service.create_prompt(
        CreatePromptIn(name="rag-qa-system", content="user-B-v3", userId="user-B")
    )

    templates_a = await service.list_versions("rag-qa-system", user_id="user-A")
    assert [t.version for t in templates_a] == [2, 1]
    assert all(t.user_id in (None, "user-A") for t in templates_a)


async def test_list_versions_without_user_id_returns_global_prompts_only() -> None:
    repo = FakePromptTemplateRepository()
    service = PromptService(repo)
    await service.create_prompt(CreatePromptIn(name="rag-qa-system", content="global-v1"))
    await service.create_prompt(
        CreatePromptIn(name="rag-qa-system", content="user-A-v2", userId="user-A")
    )
    await service.create_prompt(
        CreatePromptIn(name="rag-qa-system", content="user-B-v3", userId="user-B")
    )

    global_templates = await service.list_versions("rag-qa-system")
    assert len(global_templates) == 1
    assert global_templates[0].version == 1
    assert global_templates[0].user_id is None


async def test_list_versions_when_only_user_prompts_exist_returns_own_prompts_only() -> None:
    repo = FakePromptTemplateRepository()
    service = PromptService(repo)
    await service.create_prompt(
        CreatePromptIn(name="rag-qa-system", content="user-A-v1", userId="user-A")
    )
    await service.create_prompt(
        CreatePromptIn(name="rag-qa-system", content="user-B-v2", userId="user-B")
    )

    templates_a = await service.list_versions("rag-qa-system", user_id="user-A")
    assert len(templates_a) == 1
    assert templates_a[0].version == 1
    assert templates_a[0].user_id == "user-A"


async def test_activate_prompt_isolates_same_version_across_users_and_global() -> None:
    repo = FakePromptTemplateRepository()
    service = PromptService(repo)
    await repo.persist(
        PromptTemplate.create(name="rag-qa-system", content="global-v1", version=1, user_id=None)
    )
    await repo.persist(
        PromptTemplate.create(
            name="rag-qa-system", content="user-a-v1", version=1, user_id="user-A"
        )
    )
    await repo.persist(
        PromptTemplate.create(
            name="rag-qa-system", content="user-b-v1", version=1, user_id="user-B"
        )
    )

    activated_b = await service.activate_prompt("rag-qa-system", version=1, user_id="user-B")

    assert activated_b.is_active is True
    assert activated_b.user_id == "user-B"
    assert activated_b.content == "user-b-v1"
    global_v1 = await repo.find_by_name_and_version("rag-qa-system", 1, None)
    user_a_v1 = await repo.find_by_name_and_version("rag-qa-system", 1, "user-A")
    assert global_v1 is not None and global_v1.is_active is False
    assert user_a_v1 is not None and user_a_v1.is_active is False


async def test_delete_prompt_removes_version_and_falls_back_on_get_active() -> None:
    repo = FakePromptTemplateRepository()
    service = PromptService(repo)
    await service.create_prompt(
        CreatePromptIn(name="rag-qa-system", content="user-a-v1", userId="user-A")
    )
    await service.create_prompt(
        CreatePromptIn(name="rag-qa-system", content="user-a-v2", userId="user-A")
    )
    await service.activate_prompt("rag-qa-system", version=2, user_id="user-A")

    await service.delete_prompt("rag-qa-system", version=2, user_id="user-A")

    assert await repo.find_by_name_and_version("rag-qa-system", 2, "user-A") is None
    assert await repo.find_by_name_and_version("rag-qa-system", 1, "user-A") is not None
    active = await service.get_active_prompt("rag-qa-system", user_id="user-A")
    assert active.version == 0
    assert active.content == RAG_QA_DEFAULT_PROMPT


async def test_delete_prompt_falls_back_to_global_active_when_present() -> None:
    repo = FakePromptTemplateRepository()
    service = PromptService(repo)
    await service.create_prompt(
        CreatePromptIn(name="rag-qa-system", content="global-v1")
    )
    await service.activate_prompt("rag-qa-system", version=1)

    await service.create_prompt(
        CreatePromptIn(name="rag-qa-system", content="user-a-v2", userId="user-A")
    )
    await service.activate_prompt("rag-qa-system", version=2, user_id="user-A")

    await service.delete_prompt("rag-qa-system", version=2, user_id="user-A")

    active = await service.get_active_prompt("rag-qa-system", user_id="user-A")
    assert active.version == 1
    assert active.user_id is None
    assert active.content == "global-v1"


async def test_delete_prompt_isolates_same_version_across_users_and_global() -> None:
    repo = FakePromptTemplateRepository()
    service = PromptService(repo)
    await repo.persist(
        PromptTemplate.create(name="rag-qa-system", content="global-v1", version=1, user_id=None)
    )
    await repo.persist(
        PromptTemplate.create(
            name="rag-qa-system", content="user-a-v1", version=1, user_id="user-A"
        )
    )
    await repo.persist(
        PromptTemplate.create(
            name="rag-qa-system", content="user-b-v1", version=1, user_id="user-B"
        )
    )

    await service.delete_prompt("rag-qa-system", version=1, user_id="user-A")

    assert await repo.find_by_name_and_version("rag-qa-system", 1, "user-A") is None
    assert await repo.find_by_name_and_version("rag-qa-system", 1, None) is not None
    assert await repo.find_by_name_and_version("rag-qa-system", 1, "user-B") is not None


async def test_delete_prompt_raises_for_missing_or_other_users_version() -> None:
    repo = FakePromptTemplateRepository()
    service = PromptService(repo)
    await service.create_prompt(
        CreatePromptIn(name="rag-qa-system", content="user-a-v1", userId="user-A")
    )

    with pytest.raises(PromptTemplateNotFoundError):
        await service.delete_prompt("rag-qa-system", version=999, user_id="user-A")

    with pytest.raises(PromptTemplateNotFoundError):
        await service.delete_prompt("rag-qa-system", version=1, user_id="user-B")

    with pytest.raises(PromptTemplateNotFoundError):
        await service.delete_prompt("rag-qa-system", version=1, user_id=None)

    assert await repo.find_by_name_and_version("rag-qa-system", 1, "user-A") is not None


async def test_repository_find_by_name_and_version_queries_with_user_isolation() -> None:
    from unittest.mock import AsyncMock, MagicMock

    from ai_service.prompt.repository import PromptTemplateRepository

    repo = PromptTemplateRepository.__new__(PromptTemplateRepository)
    collection = MagicMock()
    collection.find_one = AsyncMock(return_value=None)
    repo._collection = collection

    await repo.find_by_name_and_version("rag-qa-system", 1, user_id="user-A")
    collection.find_one.assert_awaited_once_with(
        {"name": "rag-qa-system", "version": 1, "userId": "user-A"}
    )

    collection.find_one.reset_mock()
    await repo.find_by_name_and_version("rag-qa-system", 1)
    collection.find_one.assert_awaited_once_with(
        {"name": "rag-qa-system", "version": 1, "userId": {"$exists": False}}
    )


async def test_repository_delete_by_name_and_version_queries_with_user_isolation() -> None:
    from unittest.mock import AsyncMock, MagicMock

    from ai_service.prompt.repository import PromptTemplateRepository

    repo = PromptTemplateRepository.__new__(PromptTemplateRepository)
    collection = MagicMock()
    collection.delete_one = AsyncMock(return_value=MagicMock(deleted_count=1))
    repo._collection = collection

    deleted = await repo.delete_by_name_and_version("rag-qa-system", 2, user_id="user-A")
    assert deleted is True
    collection.delete_one.assert_awaited_once_with(
        {"name": "rag-qa-system", "version": 2, "userId": "user-A"}
    )

    collection.delete_one = AsyncMock(return_value=MagicMock(deleted_count=0))
    deleted_global = await repo.delete_by_name_and_version("rag-qa-system", 2)
    assert deleted_global is False
    collection.delete_one.assert_awaited_once_with(
        {"name": "rag-qa-system", "version": 2, "userId": {"$exists": False}}
    )
