from dataclasses import dataclass
from typing import Literal

from ai_service.shared_kernel.aggregate_root import AggregateRoot

BreakerStatus = Literal["closed", "open", "half-open"]

FAILURE_THRESHOLD = 5
RESET_TIMEOUT_MS = 60_000


@dataclass(frozen=True)
class CircuitBreakerSnapshot:
    model: str
    status: BreakerStatus
    failure_count: int
    opened_at: int | None


class CircuitBreakerState(AggregateRoot):
    def __init__(
        self,
        model: str,
        status: BreakerStatus,
        failure_count: int,
        opened_at: int | None,
    ) -> None:
        super().__init__()
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
