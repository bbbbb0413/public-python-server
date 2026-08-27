import logging
import time
from typing import cast

from redis.asyncio import Redis

from ai_service.llm_gateway.schemas import (
    BreakerStatus,
    CircuitBreakerSnapshot,
)

logger = logging.getLogger(__name__)

CIRCUIT_BREAKER_DB = 3
TTL_SECONDS = 3600
FAILURE_THRESHOLD = 5
RESET_TIMEOUT_MS = 60_000


def _now_ms() -> int:
    return int(time.time() * 1000)


class CircuitBreakerState:
    def __init__(
        self,
        model: str,
        status: BreakerStatus,
        failure_count: int,
        opened_at: int | None,
    ) -> None:
        self.model = model
        self._status: BreakerStatus = status
        self._failure_count = failure_count
        self._opened_at = opened_at

    @classmethod
    def create(cls, model: str) -> "CircuitBreakerState":
        return cls(model, "closed", 0, None)

    @classmethod
    def restore(cls, snapshot: CircuitBreakerSnapshot) -> "CircuitBreakerState":
        return cls(
            snapshot.model,
            snapshot.status,
            snapshot.failure_count,
            snapshot.opened_at,
        )

    def can_call(self, now_ms: int) -> bool:
        if (
            self._status == "open"
            and self._opened_at is not None
            and now_ms - self._opened_at >= RESET_TIMEOUT_MS
        ):
            self._status = "half-open"
        return self._status != "open"

    def record_failure(self, now_ms: int) -> None:
        self._failure_count += 1
        if self._failure_count >= FAILURE_THRESHOLD or self._status == "half-open":
            self._status = "open"
            self._opened_at = now_ms

    def record_success(self) -> None:
        self._status = "closed"
        self._failure_count = 0
        self._opened_at = None

    def get_status(self) -> BreakerStatus:
        return self._status

    def snapshot(self) -> CircuitBreakerSnapshot:
        return CircuitBreakerSnapshot(
            model=self.model,
            status=self._status,
            failure_count=self._failure_count,
            opened_at=self._opened_at,
        )


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


CircuitBreaker = CircuitBreakerAdapter

__all__ = [
    "CIRCUIT_BREAKER_DB",
    "CircuitBreaker",
    "CircuitBreakerAdapter",
    "CircuitBreakerState",
]
