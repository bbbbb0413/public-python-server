from typing import Annotated

from fastapi import Depends

from ai_service.config.dependencies import MongoDbDep
from ai_service.rag.application.delete_session_use_case import DeleteSessionUseCase
from ai_service.rag.application.get_session_use_case import GetSessionUseCase
from ai_service.rag.application.get_sessions_use_case import GetSessionsUseCase
from ai_service.rag.domain.repository.conversation_session_repository import (
    IConversationSessionRepository,
)
from ai_service.rag.infrastructure.persistence.conversation_session_repository_impl import (
    ConversationSessionRepositoryImpl,
)


def get_conversation_session_repository(db: MongoDbDep) -> IConversationSessionRepository:
    return ConversationSessionRepositoryImpl(db)


ConversationSessionRepositoryDep = Annotated[
    IConversationSessionRepository, Depends(get_conversation_session_repository)
]


def get_sessions_use_case(repo: ConversationSessionRepositoryDep) -> GetSessionsUseCase:
    return GetSessionsUseCase(repo)


def get_session_use_case(repo: ConversationSessionRepositoryDep) -> GetSessionUseCase:
    return GetSessionUseCase(repo)


def get_delete_session_use_case(repo: ConversationSessionRepositoryDep) -> DeleteSessionUseCase:
    return DeleteSessionUseCase(repo)


GetSessionsUseCaseDep = Annotated[GetSessionsUseCase, Depends(get_sessions_use_case)]
GetSessionUseCaseDep = Annotated[GetSessionUseCase, Depends(get_session_use_case)]
DeleteSessionUseCaseDep = Annotated[DeleteSessionUseCase, Depends(get_delete_session_use_case)]
