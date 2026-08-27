from ai_service.prompt.domain.repository.prompt_template_repository import (
    IPromptTemplateRepository,
)


class DeactivateActivePromptUseCase:
    def __init__(self, repo: IPromptTemplateRepository) -> None:
        self._repo = repo

    async def execute(self, name: str, user_id: str) -> None:
        await self._repo.deactivate_active_for_user(name, user_id)
