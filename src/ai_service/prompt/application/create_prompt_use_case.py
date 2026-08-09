from ai_service.prompt.application.command.create_prompt_command import (
    CreatePromptCommand,
)
from ai_service.prompt.domain.model.prompt_template import PromptTemplate
from ai_service.prompt.domain.repository.prompt_template_repository import (
    IPromptTemplateRepository,
)


class CreatePromptUseCase:
    def __init__(self, repo: IPromptTemplateRepository) -> None:
        self._repo = repo

    async def execute(self, command: CreatePromptCommand) -> PromptTemplate:
        existing = await self._repo.find_all_by_name(command.name)
        next_version = max((t.version for t in existing), default=0) + 1

        template = PromptTemplate.create(
            name=command.name,
            content=command.content,
            variables=command.variables,
            version=next_version,
            user_id=command.user_id,
        )
        return await self._repo.persist(template)
