from fastapi import APIRouter, Query

from ai_service.observability.dependencies import RagasEvaluationRepositoryDep
from ai_service.observability.schemas import (
    RagasEvaluationListOut,
    RagasEvaluationOut,
)

MIN_LIMIT = 1
MAX_LIMIT = 100
DEFAULT_LIMIT = 20

router = APIRouter(prefix="/observability", tags=["observability"])


@router.get("/ragas-evals")
async def get_evals(
    repo: RagasEvaluationRepositoryDep, limit: int = Query(default=DEFAULT_LIMIT)
) -> RagasEvaluationListOut:
    bounded_limit = min(max(limit, MIN_LIMIT), MAX_LIMIT)
    evaluations = await repo.find_recent(bounded_limit)
    return RagasEvaluationListOut(data=[RagasEvaluationOut.from_domain(e) for e in evaluations])


__all__ = ["router"]
