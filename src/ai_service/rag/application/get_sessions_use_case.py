from ai_service.rag.domain.model.conversation_session import ConversationSession
from ai_service.rag.domain.repository.conversation_session_repository import (
    IConversationSessionRepository,
)


class GetSessionsUseCase:
    def __init__(self, session_repo: IConversationSessionRepository) -> None:
        self._session_repo = session_repo

    async def execute(self, user_id: str, page: int, limit: int) -> list[ConversationSession]:
        return await self._session_repo.find_by_user_id(user_id, page, limit)
