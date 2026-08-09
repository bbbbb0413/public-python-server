from fastapi import APIRouter, Query

from ai_service.rag.presentation.deps import (
    DeleteSessionUseCaseDep,
    GetSessionsUseCaseDep,
    GetSessionUseCaseDep,
)
from ai_service.rag.presentation.dto import SessionDetailOut, SessionOut

DEFAULT_PAGE = 1
DEFAULT_LIMIT = 10

router = APIRouter(prefix="/rag/sessions", tags=["rag"])


@router.get("", response_model=list[SessionOut], response_model_by_alias=True)
async def get_sessions(
    use_case: GetSessionsUseCaseDep,
    user_id: str = Query(alias="userId"),
    page: int = Query(default=DEFAULT_PAGE),
    limit: int = Query(default=DEFAULT_LIMIT),
) -> list[SessionOut]:
    sessions = await use_case.execute(user_id, page, limit)
    return [SessionOut.from_domain(s) for s in sessions]


@router.get("/{session_id}", response_model=SessionDetailOut | None, response_model_by_alias=True)
async def get_session(session_id: str, use_case: GetSessionUseCaseDep) -> SessionDetailOut | None:
    session = await use_case.execute(session_id)
    if session is None:
        return None
    return SessionDetailOut.from_domain(session)


@router.delete("/{session_id}", status_code=204)
async def delete_session(session_id: str, use_case: DeleteSessionUseCaseDep) -> None:
    await use_case.execute(session_id)
