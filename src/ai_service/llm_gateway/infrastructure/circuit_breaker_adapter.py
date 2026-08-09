import logging
import time
from typing import cast

from redis.asyncio import Redis

from ai_service.llm_gateway.domain.model.circuit_breaker_state import (
    BreakerStatus,
    CircuitBreakerSnapshot,
    CircuitBreakerState,
)

logger = logging.getLogger(__name__)

CIRCUIT_BREAKER_DB = 3
TTL_SECONDS = 3600


def _now_ms() -> int:
    return int(time.time() * 1000)


class CircuitBreakerAdapter:
    def __init__(self, redis_client: Redis) -> None:
        self._redis = redis_client

    async def can_call(self, model: str) -> bool:
        state = await self._load_state(model)
        return state.can_call(_now_ms())

    async def record_success(self, model: str) -> None:
        state = await self._load_state(model)
        state.record_success()
        await self._save_state(state)

    async def record_failure(self, model: str) -> None:
        state = await self._load_state(model)
        state.record_failure(_now_ms())
        await self._save_state(state)

    async def get_state(self, model: str) -> CircuitBreakerSnapshot:
        state = await self._load_state(model)
        return state.snapshot()

    async def _load_state(self, model: str) -> CircuitBreakerState:
        try:
            raw = cast(dict[str, str], await self._redis.hgetall(f"cb:{model}"))
            if not raw or "status" not in raw:
                return CircuitBreakerState.create(model)
            status = cast(BreakerStatus, raw["status"])
            return CircuitBreakerState.restore(
                CircuitBreakerSnapshot(
                    model=model,
                    status=status,
                    failure_count=int(raw.get("failureCount", 0)),
                    opened_at=int(raw["openedAt"]) if raw.get("openedAt") else None,
                )
            )
        except Exception as e:  # noqa: BLE001 - Redis 장애 시 안전하게 closed 상태로 폴백
            logger.error("Circuit Breaker 상태 로드 실패(%s): %s", model, e)
            return CircuitBreakerState.create(model)

    async def _save_state(self, state: CircuitBreakerState) -> None:
        snap = state.snapshot()
        key = f"cb:{snap.model}"
        await self._redis.hset(
            key,
            mapping={
                "status": snap.status,
                "failureCount": str(snap.failure_count),
                "openedAt": str(snap.opened_at) if snap.opened_at is not None else "",
            },
        )
        await self._redis.expire(key, TTL_SECONDS)
