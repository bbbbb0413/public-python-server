from typing import Annotated

from fastapi import Depends

from ai_service.core.database import MongoDbDep
from ai_service.rag.repository import (
    BookmarkRepository,
    ConversationSessionRepository,
)
from ai_service.rag.service import BookmarkService, SessionService


def get_conversation_session_repository(db: MongoDbDep) -> ConversationSessionRepository:
    return ConversationSessionRepository(db)


ConversationSessionRepositoryDep = Annotated[
    ConversationSessionRepository, Depends(get_conversation_session_repository)
]


def get_session_service(repo: ConversationSessionRepositoryDep) -> SessionService:
    return SessionService(repo)


SessionServiceDep = Annotated[SessionService, Depends(get_session_service)]


def get_bookmark_repository(db: MongoDbDep) -> BookmarkRepository:
    return BookmarkRepository(db)


BookmarkRepositoryDep = Annotated[
    BookmarkRepository, Depends(get_bookmark_repository)
]


def get_bookmark_service(
    bookmark_repo: BookmarkRepositoryDep,
    session_repo: ConversationSessionRepositoryDep,
) -> BookmarkService:
    return BookmarkService(bookmark_repo=bookmark_repo, session_repo=session_repo)


BookmarkServiceDep = Annotated[BookmarkService, Depends(get_bookmark_service)]

__all__ = [
    "BookmarkRepositoryDep",
    "BookmarkServiceDep",
    "ConversationSessionRepositoryDep",
    "SessionServiceDep",
    "get_bookmark_repository",
    "get_bookmark_service",
    "get_conversation_session_repository",
    "get_session_service",
]
