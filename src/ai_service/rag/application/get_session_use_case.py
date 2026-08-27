from typing import Any

from ai_service.rag.schemas import ConversationSession


class GetSessionUseCase:
    def __init__(self, session_repo: Any) -> None:
        self._session_repo = session_repo

    async def execute(self, session_id: str) -> ConversationSession | None:
        return await self._session_repo.find_by_id(session_id)  # type: ignore[no-any-return]


__all__ = ["GetSessionUseCase"]
