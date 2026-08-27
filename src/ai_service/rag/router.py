from fastapi import APIRouter, Query

from ai_service.rag.dependencies import SessionServiceDep
from ai_service.rag.schemas import SessionDetailOut, SessionOut

DEFAULT_PAGE = 1
DEFAULT_LIMIT = 10

router = APIRouter(prefix="/rag/sessions", tags=["rag"])


@router.get("", response_model=list[SessionOut], response_model_by_alias=True)
async def get_sessions(
    service: SessionServiceDep,
    user_id: str = Query(alias="userId"),
    page: int = Query(default=DEFAULT_PAGE),
    limit: int = Query(default=DEFAULT_LIMIT),
) -> list[SessionOut]:
    sessions = await service.get_sessions(user_id, page, limit)
    return [SessionOut.from_domain(s) for s in sessions]


@router.get("/{session_id}", response_model=SessionDetailOut | None, response_model_by_alias=True)
async def get_session(session_id: str, service: SessionServiceDep) -> SessionDetailOut | None:
    session = await service.get_session(session_id)
    if session is None:
        return None
    return SessionDetailOut.from_domain(session)


@router.delete("/{session_id}", status_code=204)
async def delete_session(session_id: str, service: SessionServiceDep) -> None:
    await service.delete_session(session_id)


__all__ = ["router"]
