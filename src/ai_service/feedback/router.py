from fastapi import APIRouter, HTTPException, Query

from ai_service.feedback.dependencies import FeedbackServiceDep
from ai_service.feedback.schemas import FeedbackOut, SubmitFeedbackIn
from ai_service.feedback.service import FeedbackError

router = APIRouter(prefix="/rag/feedback", tags=["feedback"])


@router.post("", response_model=FeedbackOut, response_model_by_alias=True)
async def submit(
    dto: SubmitFeedbackIn,
    service: FeedbackServiceDep,
    user_id: str = Query(alias="userId"),
) -> FeedbackOut:
    """답변 평가 제출. 이미 평가한 답변이면 갱신한다.

    `userId` 는 게이트웨이가 인증 세션에서 채운다. 본문으로 받지 않는다 —
    받으면 남의 이름으로 평가를 남길 수 있다.
    """
    try:
        feedback = await service.submit(dto, user_id)
    except FeedbackError as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc)) from exc
    return FeedbackOut.from_domain(feedback)


@router.get("", response_model=list[FeedbackOut], response_model_by_alias=True)
async def get_for_session(
    service: FeedbackServiceDep,
    session_id: str = Query(alias="sessionId"),
    user_id: str = Query(alias="userId"),
) -> list[FeedbackOut]:
    """한 세션에서 내가 남긴 평가 전부. 다른 사용자의 평가는 돌려주지 않는다."""
    try:
        items = await service.get_for_session(session_id, user_id)
    except FeedbackError as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc)) from exc
    return [FeedbackOut.from_domain(f) for f in items]


__all__ = ["router"]
