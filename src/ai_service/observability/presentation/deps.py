from typing import Annotated

from fastapi import Depends

from ai_service.config.dependencies import MongoDbDep
from ai_service.observability.domain.repository.ragas_evaluation_repository import (
    IRagasEvaluationRepository,
)
from ai_service.observability.infrastructure.persistence.ragas_evaluation_repository_impl import (
    RagasEvaluationRepositoryImpl,
)


def get_ragas_evaluation_repository(db: MongoDbDep) -> IRagasEvaluationRepository:
    return RagasEvaluationRepositoryImpl(db)


RagasEvaluationRepositoryDep = Annotated[
    IRagasEvaluationRepository, Depends(get_ragas_evaluation_repository)
]
