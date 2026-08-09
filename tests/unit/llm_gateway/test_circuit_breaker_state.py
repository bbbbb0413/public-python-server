from ai_service.llm_gateway.domain.model.circuit_breaker_state import (
    FAILURE_THRESHOLD,
    RESET_TIMEOUT_MS,
    CircuitBreakerSnapshot,
    CircuitBreakerState,
)


def test_new_breaker_is_closed_and_callable() -> None:
    state = CircuitBreakerState.create("model-a")

    assert state.get_status() == "closed"
    assert state.can_call(now_ms=0) is True


def test_failures_below_threshold_stay_closed() -> None:
    state = CircuitBreakerState.create("model-a")

    for i in range(FAILURE_THRESHOLD - 1):
        state.record_failure(now_ms=i)

    assert state.get_status() == "closed"
    assert state.can_call(now_ms=100) is True


def test_reaching_threshold_opens_circuit() -> None:
    state = CircuitBreakerState.create("model-a")

    for i in range(FAILURE_THRESHOLD):
        state.record_failure(now_ms=i)

    assert state.get_status() == "open"
    assert state.can_call(now_ms=FAILURE_THRESHOLD) is False


def test_open_circuit_transitions_to_half_open_after_timeout() -> None:
    state = CircuitBreakerState.create("model-a")
    for _ in range(FAILURE_THRESHOLD):
        state.record_failure(now_ms=0)

    assert state.can_call(now_ms=RESET_TIMEOUT_MS - 1) is False
    assert state.can_call(now_ms=RESET_TIMEOUT_MS) is True
    assert state.get_status() == "half-open"


def test_half_open_failure_reopens_immediately() -> None:
    snapshot = CircuitBreakerSnapshot(
        model="model-a", status="half-open", failure_count=FAILURE_THRESHOLD, opened_at=0
    )
    state = CircuitBreakerState.restore(snapshot)

    state.record_failure(now_ms=1000)

    assert state.get_status() == "open"


def test_record_success_resets_state() -> None:
    state = CircuitBreakerState.create("model-a")
    for i in range(FAILURE_THRESHOLD):
        state.record_failure(now_ms=i)

    state.record_success()

    snap = state.snapshot()
    assert snap.status == "closed"
    assert snap.failure_count == 0
    assert snap.opened_at is None
