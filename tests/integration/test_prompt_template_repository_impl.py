import pytest

from ai_service.prompt.repository import PromptTemplateRepository
from ai_service.prompt.schemas import PromptTemplate

pytestmark = pytest.mark.integration


async def test_persist_assigns_id(mongo_test_db) -> None:  # type: ignore[no-untyped-def]
    repo = PromptTemplateRepository(mongo_test_db)
    template = PromptTemplate.create(name="rag-qa-system", content="c", variables=["context"])

    saved = await repo.persist(template)

    assert saved.id is not None


async def test_find_all_by_name_sorted_by_version_desc(mongo_test_db) -> None:  # type: ignore[no-untyped-def]
    repo = PromptTemplateRepository(mongo_test_db)
    await repo.persist(PromptTemplate.create(name="p", content="v1", version=1))
    await repo.persist(PromptTemplate.create(name="p", content="v2", version=2))

    results = await repo.find_all_by_name("p")

    assert [t.version for t in results] == [2, 1]


async def test_activate_flow_deactivates_previous(mongo_test_db) -> None:  # type: ignore[no-untyped-def]
    repo = PromptTemplateRepository(mongo_test_db)
    v1 = await repo.persist(PromptTemplate.create(name="p", content="v1", version=1))
    await repo.update(v1.activate())

    await repo.deactivate_all_by_name("p")
    v2 = await repo.persist(PromptTemplate.create(name="p", content="v2", version=2))
    await repo.update(v2.activate())

    active = await repo.find_active("p")
    assert active is not None
    assert active.version == 2

    reloaded_v1 = await repo.find_by_name_and_version("p", 1)
    assert reloaded_v1 is not None
    assert reloaded_v1.is_active is False


async def test_find_active_for_user(mongo_test_db) -> None:  # type: ignore[no-untyped-def]
    repo = PromptTemplateRepository(mongo_test_db)
    user_template = await repo.persist(
        PromptTemplate.create(name="p", content="for-user", version=1, user_id="user-1")
    )
    await repo.update(user_template.activate())

    found = await repo.find_active_for_user("p", "user-1")
    assert found is not None
    assert found.content == "for-user"

    not_found = await repo.find_active_for_user("p", "user-2")
    assert not_found is None


async def test_find_all_by_name_user_isolation(mongo_test_db) -> None:  # type: ignore[no-untyped-def]
    repo = PromptTemplateRepository(mongo_test_db)
    await repo.persist(PromptTemplate.create(name="iso-p", content="global-v1", version=1))
    await repo.persist(
        PromptTemplate.create(name="iso-p", content="userA-v2", version=2, user_id="user-A")
    )
    await repo.persist(
        PromptTemplate.create(name="iso-p", content="userB-v3", version=3, user_id="user-B")
    )

    # 1. user-A should only see global (v1) and user-A (v2)
    user_a_prompts = await repo.find_all_by_name("iso-p", user_id="user-A")
    assert [t.version for t in user_a_prompts] == [2, 1]

    # 2. No user_id should only see global (v1)
    global_prompts = await repo.find_all_by_name("iso-p")
    assert [t.version for t in global_prompts] == [1]

