import pytest

from ai_service.llm_gateway.circuit_breaker import (
    FAILURE_THRESHOLD,
    CircuitBreakerAdapter,
)

pytestmark = pytest.mark.integration


async def test_new_model_can_call(redis_test_client) -> None:  # type: ignore[no-untyped-def]
    adapter = CircuitBreakerAdapter(redis_test_client)

    assert await adapter.can_call("model-a") is True


async def test_failures_persist_across_instances(redis_test_client) -> None:  # type: ignore[no-untyped-def]
    adapter = CircuitBreakerAdapter(redis_test_client)

    for _ in range(FAILURE_THRESHOLD):
        await adapter.record_failure("model-a")

    reloaded = CircuitBreakerAdapter(redis_test_client)
    assert await reloaded.can_call("model-a") is False

    state = await reloaded.get_state("model-a")
    assert state.status == "open"
    assert state.failure_count == FAILURE_THRESHOLD


async def test_record_success_resets_persisted_state(redis_test_client) -> None:  # type: ignore[no-untyped-def]
    adapter = CircuitBreakerAdapter(redis_test_client)
    await adapter.record_failure("model-a")
    await adapter.record_success("model-a")

    state = await adapter.get_state("model-a")
    assert state.status == "closed"
    assert state.failure_count == 0
