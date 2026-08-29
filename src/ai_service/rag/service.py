from typing import Any

from ai_service.rag.repository import ConversationSessionRepository
from ai_service.rag.schemas import ConversationSession


class SessionNotFoundError(LookupError):
    """세션이 없거나 요청한 사람의 것이 아니다.

    두 경우를 구분하지 않는 것이 의도다. 나누어 답하면 세션 id 하나로
    "그런 세션이 있긴 하다" 를 알아낼 수 있다.
    """


class SessionService:
    def __init__(self, repo: ConversationSessionRepository | Any) -> None:
        self._repo = repo

    async def get_sessions(
        self, user_id: str, page: int, limit: int
    ) -> list[ConversationSession]:
        return await self._repo.find_by_user_id(user_id, page, limit)

    async def get_session(self, session_id: str, user_id: str) -> ConversationSession:
        session = await self._repo.find_by_id_for_user(session_id, user_id)
        if session is None:
            raise SessionNotFoundError(session_id)
        return session

    async def delete_session(self, session_id: str, user_id: str) -> None:
        deleted = await self._repo.delete_by_id_for_user(session_id, user_id)
        if not deleted:
            raise SessionNotFoundError(session_id)


__all__ = ["SessionNotFoundError", "SessionService"]
