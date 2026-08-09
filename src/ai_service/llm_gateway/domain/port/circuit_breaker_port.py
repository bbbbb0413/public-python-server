from typing import Protocol

from ai_service.llm_gateway.domain.model.circuit_breaker_state import (
    CircuitBreakerSnapshot,
)


class ICircuitBreakerPort(Protocol):
    async def can_call(self, model: str) -> bool: ...

    async def record_success(self, model: str) -> None: ...

    async def record_failure(self, model: str) -> None: ...

    async def get_state(self, model: str) -> CircuitBreakerSnapshot: ...
