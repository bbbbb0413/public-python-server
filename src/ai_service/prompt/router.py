from fastapi import APIRouter, HTTPException, Query

from ai_service.prompt.dependencies import PromptServiceDep
from ai_service.prompt.schemas import (
    CreatePromptIn,
    PromptOut,
    PromptTemplateNotFoundError,
)

router = APIRouter(prefix="/prompts", tags=["prompts"])


@router.post("", response_model=PromptOut, response_model_by_alias=True)
async def create(dto: CreatePromptIn, service: PromptServiceDep) -> PromptOut:
    template = await service.create_prompt(dto)
    return PromptOut.from_domain(template)


@router.get("/{name}", response_model=list[PromptOut], response_model_by_alias=True)
async def list_versions(
    name: str,
    service: PromptServiceDep,
    user_id: str | None = Query(default=None, alias="userId"),
) -> list[PromptOut]:
    templates = await service.list_versions(name, user_id=user_id)
    return [PromptOut.from_domain(t) for t in templates]


@router.get("/{name}/active", response_model=PromptOut, response_model_by_alias=True)
async def get_active(
    name: str,
    service: PromptServiceDep,
    user_id: str | None = Query(default=None, alias="userId"),
) -> PromptOut:
    template = await service.get_active_prompt(name, user_id)
    return PromptOut.from_domain(template)


@router.patch("/{name}/{version}/activate", response_model=PromptOut, response_model_by_alias=True)
async def activate(
    name: str,
    version: int,
    service: PromptServiceDep,
    user_id: str | None = Query(default=None, alias="userId"),
) -> PromptOut:
    try:
        template = await service.activate_prompt(name, version, user_id)
    except PromptTemplateNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    return PromptOut.from_domain(template)


@router.delete("/{name}/active", status_code=204)
async def reset_active_for_user(
    name: str,
    service: PromptServiceDep,
    user_id: str = Query(..., alias="userId"),
) -> None:
    await service.deactivate_active_prompt(name, user_id)


__all__ = ["router"]
