from typing import Annotated

from fastapi import Depends

from ai_service.core.database import MongoDbDep
from ai_service.prompt.repository import PromptTemplateRepository
from ai_service.prompt.service import PromptService


def get_prompt_template_repository(db: MongoDbDep) -> PromptTemplateRepository:
    return PromptTemplateRepository(db)


PromptTemplateRepositoryDep = Annotated[
    PromptTemplateRepository, Depends(get_prompt_template_repository)
]


def get_prompt_service(repo: PromptTemplateRepositoryDep) -> PromptService:
    return PromptService(repo)


PromptServiceDep = Annotated[PromptService, Depends(get_prompt_service)]

__all__ = [
    "PromptServiceDep",
    "PromptTemplateRepositoryDep",
    "get_prompt_service",
    "get_prompt_template_repository",
]
