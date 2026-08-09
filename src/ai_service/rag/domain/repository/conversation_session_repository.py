from typing import Protocol

from ai_service.rag.domain.model.conversation_session import ConversationSession


class IConversationSessionRepository(Protocol):
    async def find_by_id(self, session_id: str) -> ConversationSession | None: ...

    async def find_by_user_id(
        self, user_id: str, page: int, limit: int
    ) -> list[ConversationSession]: ...

    async def persist(self, session: ConversationSession) -> ConversationSession: ...

    async def update(self, session: ConversationSession) -> ConversationSession: ...

    async def delete_by_id(self, session_id: str) -> None: ...
