from typing import Any

from ai_service.rag.schemas import ConversationSession


class GetSessionsUseCase:
    def __init__(self, session_repo: Any) -> None:
        self._session_repo = session_repo

    async def execute(self, user_id: str, page: int, limit: int) -> list[ConversationSession]:
        return await self._session_repo.find_by_user_id(user_id, page, limit)  # type: ignore[no-any-return]


__all__ = ["GetSessionsUseCase"]
