from ai_service.prompt.application.command.activate_prompt_command import (
    ActivatePromptCommand,
)
from ai_service.prompt.application.exceptions import PromptTemplateNotFoundError
from ai_service.prompt.domain.model.prompt_template import PromptTemplate
from ai_service.prompt.domain.repository.prompt_template_repository import (
    IPromptTemplateRepository,
)


class ActivatePromptUseCase:
    def __init__(self, repo: IPromptTemplateRepository) -> None:
        self._repo = repo

    async def execute(self, command: ActivatePromptCommand) -> PromptTemplate:
        target = await self._repo.find_by_name_and_version(command.name, command.version)
        if target is None:
            raise PromptTemplateNotFoundError(command.name, command.version)

        await self._repo.deactivate_all_by_name(command.name)
        activated = target.activate()
        return await self._repo.update(activated)
