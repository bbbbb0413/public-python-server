from ai_service.prompt.application.command.create_prompt_command import (
    CreatePromptCommand,
)
from ai_service.prompt.application.create_prompt_use_case import CreatePromptUseCase
from tests.unit.prompt.fakes import FakePromptTemplateRepository


async def test_first_version_starts_at_one() -> None:
    repo = FakePromptTemplateRepository()
    use_case = CreatePromptUseCase(repo)

    template = await use_case.execute(
        CreatePromptCommand(name="rag-qa-system", content="c", variables=["context"])
    )

    assert template.version == 1


async def test_next_version_increments_from_existing_max() -> None:
    repo = FakePromptTemplateRepository()
    use_case = CreatePromptUseCase(repo)
    await use_case.execute(CreatePromptCommand(name="rag-qa-system", content="v1"))
    await use_case.execute(CreatePromptCommand(name="rag-qa-system", content="v2"))

    template = await use_case.execute(CreatePromptCommand(name="rag-qa-system", content="v3"))

    assert template.version == 3
