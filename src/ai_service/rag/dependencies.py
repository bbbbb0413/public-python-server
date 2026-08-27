from typing import Annotated

from fastapi import Depends

from ai_service.core.database import MongoDbDep
from ai_service.rag.repository import ConversationSessionRepository
from ai_service.rag.service import SessionService


def get_conversation_session_repository(db: MongoDbDep) -> ConversationSessionRepository:
    return ConversationSessionRepository(db)


ConversationSessionRepositoryDep = Annotated[
    ConversationSessionRepository, Depends(get_conversation_session_repository)
]


def get_session_service(repo: ConversationSessionRepositoryDep) -> SessionService:
    return SessionService(repo)


SessionServiceDep = Annotated[SessionService, Depends(get_session_service)]

__all__ = [
    "ConversationSessionRepositoryDep",
    "SessionServiceDep",
    "get_conversation_session_repository",
    "get_session_service",
]
