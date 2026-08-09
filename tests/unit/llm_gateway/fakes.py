from collections.abc import AsyncIterator
from datetime import datetime

from ai_service.llm_gateway.domain.model.circuit_breaker_state import CircuitBreakerSnapshot
from ai_service.llm_gateway.domain.model.llm_message import LlmMessage, LlmOptions
from ai_service.llm_gateway.domain.repository.llm_cost_log_repository import (
    LlmCostLog,
    ModelCostSum,
)


class FakeLlmProvider:
    def __init__(self, behavior: dict[str, list[str] | Exception]) -> None:
        self._behavior = behavior

    async def chat(self, messages: list[LlmMessage], options: LlmOptions | None = None) -> str:
        raise NotImplementedError

    async def stream(
        self, messages: list[LlmMessage], options: LlmOptions | None = None
    ) -> AsyncIterator[str]:
        model = options.model if options and options.model else "default"
        result = self._behavior[model]
        if isinstance(result, Exception):
            raise result
        for token in result:
            yield token


class FakeCircuitBreaker:
    def __init__(self, blocked: set[str] | None = None) -> None:
        self.blocked = blocked or set()
        self.successes: list[str] = []
        self.failures: list[str] = []

    async def can_call(self, model: str) -> bool:
        return model not in self.blocked

    async def record_success(self, model: str) -> None:
        self.successes.append(model)

    async def record_failure(self, model: str) -> None:
        self.failures.append(model)

    async def get_state(self, model: str) -> CircuitBreakerSnapshot:
        return CircuitBreakerSnapshot(model=model, status="closed", failure_count=0, opened_at=None)


class FakeLlmCostLogRepository:
    def __init__(self) -> None:
        self.logs: list[LlmCostLog] = []

    async def persist(self, log: LlmCostLog) -> None:
        self.logs.append(log)

    async def sum_by_model(self, from_: datetime, to: datetime) -> list[ModelCostSum]:
        return []


class FailingLlmCostLogRepository:
    async def persist(self, log: LlmCostLog) -> None:
        raise ConnectionError("mongo down")

    async def sum_by_model(self, from_: datetime, to: datetime) -> list[ModelCostSum]:
        return []
