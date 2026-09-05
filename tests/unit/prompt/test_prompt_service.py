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
    await repo.persist(active)

    service = PromptService(repo)
    template = await service.get_active_prompt("rag-qa-system")

    assert template.content == "global"


async def test_prefers_user_specific_active_over_global() -> None:
    repo = FakePromptTemplateRepository()
    global_active = PromptTemplate.create(name="rag-qa-system", content="global").activate()
    await repo.persist(global_active)

    user_active = PromptTemplate.create(
        name="rag-qa-system", content="for-user", version=2, user_id="user-1"
    ).activate()
    await repo.persist(user_active)

    service = PromptService(repo)
    template = await service.get_active_prompt("rag-qa-system", user_id="user-1")

    assert template.content == "for-user"


async def test_other_users_active_prompt_never_leaks_as_fallback() -> None:
    repo = FakePromptTemplateRepository()
    user1_active = PromptTemplate.create(
        name="rag-qa-system", content="user-1-only", user_id="user-1"
    ).activate()
    await repo.persist(user1_active)

    service = PromptService(repo)
    template = await service.get_active_prompt("rag-qa-system", user_id="user-2")

    assert template.content != "user-1-only"
    assert template.content == RAG_QA_DEFAULT_PROMPT


async def test_list_versions_with_user_id_returns_user_and_global_templates_only() -> None:
    repo = FakePromptTemplateRepository()
    service = PromptService(repo)
    await service.create_prompt(CreatePromptIn(name="rag-qa-system", content="global-v1"))
    await service.create_prompt(
        CreatePromptIn(name="rag-qa-system", content="userA-v2", userId="user-A")
    )
    await service.create_prompt(
        CreatePromptIn(name="rag-qa-system", content="userB-v3", userId="user-B")
    )

    templates = await service.list_versions("rag-qa-system", user_id="user-A")
    versions = [t.version for t in templates]
    assert len(templates) == 2
    assert 2 in versions  # user-A
    assert 1 in versions  # global
    assert 3 not in versions  # user-B


async def test_list_versions_without_user_id_returns_only_global_templates() -> None:
    repo = FakePromptTemplateRepository()
    service = PromptService(repo)
    await service.create_prompt(CreatePromptIn(name="rag-qa-system", content="global-v1"))
    await service.create_prompt(
        CreatePromptIn(name="rag-qa-system", content="userA-v2", userId="user-A")
    )
    await service.create_prompt(
        CreatePromptIn(name="rag-qa-system", content="userB-v3", userId="user-B")
    )

    templates = await service.list_versions("rag-qa-system")
    assert len(templates) == 1
    assert templates[0].version == 1
    assert templates[0].user_id is None

