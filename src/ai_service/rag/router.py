from fastapi import APIRouter, HTTPException, Query

from ai_service.rag.dependencies import SessionServiceDep
from ai_service.rag.schemas import (
    SessionBookmarkOut,
    SessionDetailOut,
    SessionOut,
    ToggleBookmarkIn,
)
from ai_service.rag.service import SessionNotFoundError

DEFAULT_PAGE = 1
DEFAULT_LIMIT = 10

# 없는 세션과 남의 세션에 같은 응답을 준다. 나누면 세션 id 로 존재 여부를 알아낼 수 있다.
NOT_FOUND_DETAIL = "대화 세션을 찾을 수 없습니다."

router = APIRouter(prefix="/rag/sessions", tags=["rag"])


@router.get("", response_model=list[SessionOut], response_model_by_alias=True)
async def get_sessions(
    service: SessionServiceDep,
    user_id: str = Query(alias="userId"),
    page: int = Query(default=DEFAULT_PAGE),
    limit: int = Query(default=DEFAULT_LIMIT),
    keyword: str | None = Query(default=None),
    q: str | None = Query(default=None),
) -> list[SessionOut]:
    effective_keyword = keyword if keyword is not None else q
    sessions = await service.get_sessions(user_id, page, limit, effective_keyword)
    return [SessionOut.from_domain(s) for s in sessions]


@router.get("/{session_id}", response_model=SessionDetailOut, response_model_by_alias=True)
async def get_session(
    session_id: str,
    service: SessionServiceDep,
    user_id: str = Query(alias="userId"),
) -> SessionDetailOut:
    """대화 세션 상세. 소유자만 읽을 수 있다.

    `userId` 는 게이트웨이가 인증 세션에서 채운다.
    """
    try:
        session = await service.get_session(session_id, user_id)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=NOT_FOUND_DETAIL) from exc
    return SessionDetailOut.from_domain(session)


@router.patch(
    "/{session_id}/bookmark",
    response_model=SessionBookmarkOut,
    response_model_by_alias=True,
)
async def update_session_bookmark(
    session_id: str,
    service: SessionServiceDep,
    user_id: str = Query(alias="userId"),
    body: ToggleBookmarkIn | None = None,
) -> SessionBookmarkOut:
    """대화 세션 북마크 상태 변경(또는 토글). 소유자만 변경할 수 있다."""
    bookmarked = body.bookmarked if body is not None else None
    try:
        updated = await service.set_bookmark(session_id, user_id, bookmarked)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=NOT_FOUND_DETAIL) from exc
    return SessionBookmarkOut(
        sessionId=updated.get_session_id(),
        bookmarked=updated.is_bookmarked,
    )


@router.post(
    "/{session_id}/bookmark",
    response_model=SessionBookmarkOut,
    response_model_by_alias=True,
)
async def bookmark_session(
    session_id: str,
    service: SessionServiceDep,
    user_id: str = Query(alias="userId"),
    body: ToggleBookmarkIn | None = None,
) -> SessionBookmarkOut:
    """대화 세션 북마크 등록. 소유자만 변경할 수 있다."""
    bookmarked = (
        body.bookmarked
        if body is not None and body.bookmarked is not None
        else True
    )
    try:
        updated = await service.set_bookmark(session_id, user_id, bookmarked)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=NOT_FOUND_DETAIL) from exc
    return SessionBookmarkOut(
        sessionId=updated.get_session_id(),
        bookmarked=updated.is_bookmarked,
    )


@router.delete("/{session_id}/bookmark", status_code=204)
async def unbookmark_session(
    session_id: str,
    service: SessionServiceDep,
    user_id: str = Query(alias="userId"),
) -> None:
    """대화 세션 북마크 해제. 소유자만 변경할 수 있다."""
    try:
        await service.set_bookmark(session_id, user_id, False)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=NOT_FOUND_DETAIL) from exc


@router.delete("/{session_id}", status_code=204)
async def delete_session(
    session_id: str,
    service: SessionServiceDep,
    user_id: str = Query(alias="userId"),
) -> None:
    """대화 세션 삭제. 소유자만 지울 수 있다."""
    try:
        await service.delete_session(session_id, user_id)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=NOT_FOUND_DETAIL) from exc


__all__ = ["router"]
