from typing import Annotated

from fastapi import Depends

from ai_service.config.dependencies import CircuitBreakerRedisClientDep, MongoDbDep
from ai_service.llm_gateway.domain.port.circuit_breaker_port import ICircuitBreakerPort
from ai_service.llm_gateway.domain.repository.llm_cost_log_repository import (
    ILlmCostLogRepository,
)
from ai_service.llm_gateway.infrastructure.circuit_breaker_adapter import (
    CircuitBreakerAdapter,
)
from ai_service.llm_gateway.infrastructure.persistence.llm_cost_log_repository_impl import (
    LlmCostLogRepositoryImpl,
)


def get_llm_cost_log_repository(db: MongoDbDep) -> ILlmCostLogRepository:
    return LlmCostLogRepositoryImpl(db)


def get_circuit_breaker(redis_client: CircuitBreakerRedisClientDep) -> ICircuitBreakerPort:
    return CircuitBreakerAdapter(redis_client)


LlmCostLogRepositoryDep = Annotated[ILlmCostLogRepository, Depends(get_llm_cost_log_repository)]
CircuitBreakerDep = Annotated[ICircuitBreakerPort, Depends(get_circuit_breaker)]
