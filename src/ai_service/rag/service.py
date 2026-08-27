from typing import Any

from ai_service.rag.repository import ConversationSessionRepository
from ai_service.rag.schemas import ConversationSession


class SessionService:
    def __init__(self, repo: ConversationSessionRepository | Any) -> None:
        self._repo = repo

    async def get_sessions(
        self, user_id: str, page: int, limit: int
    ) -> list[ConversationSession]:
        return await self._repo.find_by_user_id(user_id, page, limit)

    async def get_session(self, session_id: str) -> ConversationSession | None:
        return await self._repo.find_by_id(session_id)

    async def delete_session(self, session_id: str) -> None:
        await self._repo.delete_by_id(session_id)


__all__ = ["SessionService"]
