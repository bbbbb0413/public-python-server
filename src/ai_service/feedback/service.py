from typing import Any

from ai_service.feedback.repository import AnswerFeedbackRepository
from ai_service.feedback.schemas import AnswerFeedback, SubmitFeedbackIn
from ai_service.rag.repository import ConversationSessionRepository


class FeedbackError(ValueError):
    """평가를 받을 수 없다. 라우터가 HTTP 상태로 옮긴다."""

    def __init__(self, message: str, status: int) -> None:
        super().__init__(message)
        self.status = status


class FeedbackService:
    def __init__(
        self,
        repo: AnswerFeedbackRepository | Any,
        session_repo: ConversationSessionRepository | Any,
    ) -> None:
        self._repo = repo
        self._session_repo = session_repo

    async def submit(self, dto: SubmitFeedbackIn, user_id: str) -> AnswerFeedback:
        await self._assert_answer_is_readable(dto.session_id, dto.turn_index, user_id)
        feedback = AnswerFeedback.of(
            session_id=dto.session_id,
            turn_index=dto.turn_index,
            user_id=user_id,
            accuracy=dto.accuracy,
            helpfulness=dto.helpfulness,
            comment=dto.comment,
        )
        return await self._repo.upsert(feedback)

    async def get_for_session(self, session_id: str, user_id: str) -> list[AnswerFeedback]:
        await self._assert_session_is_readable(session_id, user_id)
        return await self._repo.find_by_session(session_id, user_id)

    async def _assert_session_is_readable(self, session_id: str, user_id: str) -> Any:
        """세션 소유자만 통과시킨다.

        화면에서 버튼을 숨기는 것으로는 부족하다. 세션 id 는 추측할 수 없는
        값이지만 한 번 새어 나가면 그것만으로 남의 대화에 평가를 남길 수 있다.
        없는 세션과 남의 세션을 같은 404 로 답해 존재 여부를 흘리지 않는다.

        소유권 조건은 리포지토리 쿼리가 건다 — 읽어 온 뒤 여기서 비교하면
        호출부가 늘 때마다 빠뜨릴 수 있다.
        """
        session = await self._session_repo.find_by_id_for_user(session_id, user_id)
        if session is None:
            raise FeedbackError("대상 세션을 찾을 수 없습니다.", 404)
        return session

    async def _assert_answer_is_readable(
        self, session_id: str, turn_index: int, user_id: str
    ) -> None:
        session = await self._assert_session_is_readable(session_id, user_id)

        turns = session.turns
        if turn_index >= len(turns):
            raise FeedbackError("대상 답변을 찾을 수 없습니다.", 404)
        if turns[turn_index].role != "assistant":
            raise FeedbackError("답변이 아닌 턴에는 평가를 남길 수 없습니다.", 400)


__all__ = ["FeedbackError", "FeedbackService"]
