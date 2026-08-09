from ai_service.rag.domain.model.conversation_session import ConversationSession
from ai_service.rag.domain.repository.conversation_session_repository import (
    IConversationSessionRepository,
)


class GetSessionUseCase:
    def __init__(self, session_repo: IConversationSessionRepository) -> None:
        self._session_repo = session_repo

    async def execute(self, session_id: str) -> ConversationSession | None:
        return await self._session_repo.find_by_id(session_id)
