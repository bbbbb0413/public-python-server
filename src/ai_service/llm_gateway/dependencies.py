from typing import Annotated, Any

from fastapi import Depends

from ai_service.core.database import CircuitBreakerRedisClientDep, MongoDbDep
from ai_service.llm_gateway.circuit_breaker import CircuitBreakerAdapter
from ai_service.llm_gateway.repository import LlmCostLogRepository


def get_llm_cost_log_repository(db: MongoDbDep) -> LlmCostLogRepository:
    return LlmCostLogRepository(db)


def get_circuit_breaker(redis_client: CircuitBreakerRedisClientDep) -> CircuitBreakerAdapter:
    return CircuitBreakerAdapter(redis_client)


LlmCostLogRepositoryDep = Annotated[LlmCostLogRepository, Depends(get_llm_cost_log_repository)]
CircuitBreakerDep = Annotated[Any, Depends(get_circuit_breaker)]

__all__ = [
    "CircuitBreakerDep",
    "LlmCostLogRepositoryDep",
    "get_circuit_breaker",
    "get_llm_cost_log_repository",
]
