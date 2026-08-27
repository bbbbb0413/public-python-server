from typing import Annotated

from fastapi import Depends

from ai_service.core.config import Settings, get_settings
from ai_service.core.database import MongoDbDep
from ai_service.observability.repository import RagasEvaluationRepository
from ai_service.observability.service import RagasEvalService


def get_ragas_evaluation_repository(db: MongoDbDep) -> RagasEvaluationRepository:
    return RagasEvaluationRepository(db)


RagasEvaluationRepositoryDep = Annotated[
    RagasEvaluationRepository, Depends(get_ragas_evaluation_repository)
]


def get_ragas_eval_service(
    repo: RagasEvaluationRepositoryDep,
    settings: Annotated[Settings, Depends(get_settings)],
) -> RagasEvalService:
    return RagasEvalService(repo, settings)


RagasEvalServiceDep = Annotated[RagasEvalService, Depends(get_ragas_eval_service)]

__all__ = [
    "RagasEvalServiceDep",
    "RagasEvaluationRepositoryDep",
    "get_ragas_eval_service",
    "get_ragas_evaluation_repository",
]
