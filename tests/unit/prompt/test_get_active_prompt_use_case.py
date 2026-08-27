from ai_service.prompt.application.get_active_prompt_use_case import (
    RAG_QA_DEFAULT_PROMPT,
    GetActivePromptUseCase,
)
from ai_service.prompt.domain.model.prompt_template import PromptTemplate
from tests.unit.prompt.fakes import FakePromptTemplateRepository


async def test_returns_hardcoded_default_when_nothing_stored() -> None:
    repo = FakePromptTemplateRepository()
    use_case = GetActivePromptUseCase(repo)

    template = await use_case.execute("rag-qa-system")

    assert template.content == RAG_QA_DEFAULT_PROMPT
    assert template.version == 0
    assert template.is_active is False


async def test_prefers_global_active_over_default() -> None:
    repo = FakePromptTemplateRepository()
    active = PromptTemplate.create(name="rag-qa-system", content="global").activate()
    repo.storage[("rag-qa-system", active.version)] = active

    use_case = GetActivePromptUseCase(repo)
    template = await use_case.execute("rag-qa-system")

    assert template.content == "global"


async def test_prefers_user_specific_active_over_global() -> None:
    repo = FakePromptTemplateRepository()
    global_active = PromptTemplate.create(name="rag-qa-system", content="global").activate()
    repo.storage[("rag-qa-system", global_active.version)] = global_active

    user_active = PromptTemplate.create(
        name="rag-qa-system", content="for-user", version=2, user_id="user-1"
    ).activate()
    repo.storage[("rag-qa-system", user_active.version)] = user_active

    use_case = GetActivePromptUseCase(repo)
    template = await use_case.execute("rag-qa-system", user_id="user-1")

    assert template.content == "for-user"


async def test_other_users_active_prompt_never_leaks_as_fallback() -> None:
    repo = FakePromptTemplateRepository()
    user1_active = PromptTemplate.create(
        name="rag-qa-system", content="user-1-only", user_id="user-1"
    ).activate()
    repo.storage[("rag-qa-system", user1_active.version)] = user1_active

    use_case = GetActivePromptUseCase(repo)
    template = await use_case.execute("rag-qa-system", user_id="user-2")

    assert template.content != "user-1-only"
    assert template.content == RAG_QA_DEFAULT_PROMPT
