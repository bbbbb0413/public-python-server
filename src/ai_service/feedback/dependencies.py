from typing import Annotated

from fastapi import Depends

from ai_service.core.database import MongoDbDep
from ai_service.feedback.repository import AnswerFeedbackRepository
from ai_service.feedback.service import FeedbackService
from ai_service.rag.dependencies import ConversationSessionRepositoryDep


def get_answer_feedback_repository(db: MongoDbDep) -> AnswerFeedbackRepository:
    return AnswerFeedbackRepository(db)


AnswerFeedbackRepositoryDep = Annotated[
    AnswerFeedbackRepository, Depends(get_answer_feedback_repository)
]


def get_feedback_service(
    repo: AnswerFeedbackRepositoryDep,
    session_repo: ConversationSessionRepositoryDep,
) -> FeedbackService:
    return FeedbackService(repo, session_repo)


FeedbackServiceDep = Annotated[FeedbackService, Depends(get_feedback_service)]

__all__ = [
    "AnswerFeedbackRepositoryDep",
    "FeedbackServiceDep",
    "get_answer_feedback_repository",
    "get_feedback_service",
]
