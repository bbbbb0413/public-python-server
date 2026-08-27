import pytest

from ai_service.prompt.application.activate_prompt_use_case import ActivatePromptUseCase
from ai_service.prompt.application.command.activate_prompt_command import (
    ActivatePromptCommand,
)
from ai_service.prompt.application.command.create_prompt_command import (
    CreatePromptCommand,
)
from ai_service.prompt.application.create_prompt_use_case import CreatePromptUseCase
from ai_service.prompt.application.exceptions import PromptTemplateNotFoundError
from tests.unit.prompt.fakes import FakePromptTemplateRepository


async def test_activating_missing_version_raises() -> None:
    repo = FakePromptTemplateRepository()
    use_case = ActivatePromptUseCase(repo)

    with pytest.raises(PromptTemplateNotFoundError):
        await use_case.execute(ActivatePromptCommand(name="rag-qa-system", version=1))


async def test_activating_deactivates_previous_active_version() -> None:
    repo = FakePromptTemplateRepository()
    create_use_case = CreatePromptUseCase(repo)
    await create_use_case.execute(CreatePromptCommand(name="rag-qa-system", content="v1"))
    await create_use_case.execute(CreatePromptCommand(name="rag-qa-system", content="v2"))

    activate_use_case = ActivatePromptUseCase(repo)
    await activate_use_case.execute(ActivatePromptCommand(name="rag-qa-system", version=1))
    activated_v2 = await activate_use_case.execute(
        ActivatePromptCommand(name="rag-qa-system", version=2)
    )

    v1 = await repo.find_by_name_and_version("rag-qa-system", 1)
    assert v1 is not None
    assert v1.is_active is False
    assert activated_v2.is_active is True


async def test_user_scoped_activate_does_not_touch_global_active_version() -> None:
    repo = FakePromptTemplateRepository()
    create_use_case = CreatePromptUseCase(repo)
    await create_use_case.execute(CreatePromptCommand(name="rag-qa-system", content="global-v1"))
    await create_use_case.execute(
        CreatePromptCommand(name="rag-qa-system", content="user-v1", user_id="user-1")
    )

    activate_use_case = ActivatePromptUseCase(repo)
    global_active = await activate_use_case.execute(
        ActivatePromptCommand(name="rag-qa-system", version=1)
    )
    user_active = await activate_use_case.execute(
        ActivatePromptCommand(name="rag-qa-system", version=2, user_id="user-1")
    )

    reloaded_global = await repo.find_by_name_and_version("rag-qa-system", 1)
    assert reloaded_global is not None
    assert reloaded_global.is_active is True
    assert global_active.is_active is True
    assert user_active.is_active is True


async def test_cannot_activate_another_users_version() -> None:
    repo = FakePromptTemplateRepository()
    create_use_case = CreatePromptUseCase(repo)
    await create_use_case.execute(
        CreatePromptCommand(name="rag-qa-system", content="user-v1", user_id="user-1")
    )

    activate_use_case = ActivatePromptUseCase(repo)
    with pytest.raises(PromptTemplateNotFoundError):
        await activate_use_case.execute(
            ActivatePromptCommand(name="rag-qa-system", version=1, user_id="user-2")
        )
    with pytest.raises(PromptTemplateNotFoundError):
        await activate_use_case.execute(
            ActivatePromptCommand(name="rag-qa-system", version=1)
        )
