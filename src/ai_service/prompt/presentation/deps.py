from typing import Annotated

from fastapi import Depends

from ai_service.config.dependencies import MongoDbDep
from ai_service.prompt.application.activate_prompt_use_case import ActivatePromptUseCase
from ai_service.prompt.application.create_prompt_use_case import CreatePromptUseCase
from ai_service.prompt.application.get_active_prompt_use_case import (
    GetActivePromptUseCase,
)
from ai_service.prompt.domain.repository.prompt_template_repository import (
    IPromptTemplateRepository,
)
from ai_service.prompt.infrastructure.persistence.prompt_template_repository_impl import (
    PromptTemplateRepositoryImpl,
)


def get_prompt_template_repository(db: MongoDbDep) -> IPromptTemplateRepository:
    return PromptTemplateRepositoryImpl(db)


PromptTemplateRepositoryDep = Annotated[
    IPromptTemplateRepository, Depends(get_prompt_template_repository)
]


def get_create_prompt_use_case(repo: PromptTemplateRepositoryDep) -> CreatePromptUseCase:
    return CreatePromptUseCase(repo)


def get_activate_prompt_use_case(repo: PromptTemplateRepositoryDep) -> ActivatePromptUseCase:
    return ActivatePromptUseCase(repo)


def get_active_prompt_use_case(repo: PromptTemplateRepositoryDep) -> GetActivePromptUseCase:
    return GetActivePromptUseCase(repo)


CreatePromptUseCaseDep = Annotated[CreatePromptUseCase, Depends(get_create_prompt_use_case)]
ActivatePromptUseCaseDep = Annotated[ActivatePromptUseCase, Depends(get_activate_prompt_use_case)]
GetActivePromptUseCaseDep = Annotated[GetActivePromptUseCase, Depends(get_active_prompt_use_case)]
