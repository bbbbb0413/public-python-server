from fastapi import APIRouter, HTTPException, Query

from ai_service.prompt.application.command.activate_prompt_command import (
    ActivatePromptCommand,
)
from ai_service.prompt.application.command.create_prompt_command import (
    CreatePromptCommand,
)
from ai_service.prompt.application.exceptions import PromptTemplateNotFoundError
from ai_service.prompt.presentation.deps import (
    ActivatePromptUseCaseDep,
    CreatePromptUseCaseDep,
    GetActivePromptUseCaseDep,
    PromptTemplateRepositoryDep,
)
from ai_service.prompt.presentation.dto import CreatePromptIn, PromptOut

router = APIRouter(prefix="/prompts", tags=["prompts"])


@router.post("", response_model=PromptOut, response_model_by_alias=True)
async def create(dto: CreatePromptIn, use_case: CreatePromptUseCaseDep) -> PromptOut:
    template = await use_case.execute(
        CreatePromptCommand(
            name=dto.name, content=dto.content, variables=dto.variables, user_id=dto.user_id
        )
    )
    return PromptOut.from_domain(template)


@router.get("/{name}", response_model=list[PromptOut], response_model_by_alias=True)
async def list_versions(name: str, repo: PromptTemplateRepositoryDep) -> list[PromptOut]:
    templates = await repo.find_all_by_name(name)
    return [PromptOut.from_domain(t) for t in templates]


@router.get("/{name}/active", response_model=PromptOut, response_model_by_alias=True)
async def get_active(
    name: str, use_case: GetActivePromptUseCaseDep, user_id: str | None = Query(default=None)
) -> PromptOut:
    template = await use_case.execute(name, user_id)
    return PromptOut.from_domain(template)


@router.patch("/{name}/{version}/activate", response_model=PromptOut, response_model_by_alias=True)
async def activate(name: str, version: int, use_case: ActivatePromptUseCaseDep) -> PromptOut:
    try:
        template = await use_case.execute(ActivatePromptCommand(name=name, version=version))
    except PromptTemplateNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    return PromptOut.from_domain(template)
