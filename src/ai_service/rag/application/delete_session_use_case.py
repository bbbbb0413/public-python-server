from ai_service.rag.domain.repository.conversation_session_repository import (
    IConversationSessionRepository,
)


class DeleteSessionUseCase:
    def __init__(self, session_repo: IConversationSessionRepository) -> None:
        self._session_repo = session_repo

    async def execute(self, session_id: str) -> None:
        await self._session_repo.delete_by_id(session_id)
