import pytest

from ai_service.llm_gateway.application.fallback_service import (
    AllFallbacksFailedError,
    FallbackService,
)
from ai_service.llm_gateway.domain.model.llm_message import LlmMessage
from tests.unit.llm_gateway.fakes import FakeCircuitBreaker, FakeLlmProvider


async def _collect(service: FallbackService, chain: list[str]) -> list[tuple[str, str | None]]:
    messages = [LlmMessage(role="user", content="hi")]
    return [
        (chunk.model, chunk.token) async for chunk in service.stream_with_fallback(messages, chain)
    ]


async def test_first_model_success_short_circuits() -> None:
    llm = FakeLlmProvider({"a": ["h", "i"]})
    breaker = FakeCircuitBreaker()
    service = FallbackService(llm, breaker)

    result = await _collect(service, ["a", "b"])

    assert result == [("a", "h"), ("a", "i")]
    assert breaker.successes == ["a"]
    assert breaker.failures == []


async def test_skips_model_blocked_by_open_breaker() -> None:
    llm = FakeLlmProvider({"b": ["ok"]})
    breaker = FakeCircuitBreaker(blocked={"a"})
    service = FallbackService(llm, breaker)

    result = await _collect(service, ["a", "b"])

    assert result == [("b", "ok")]


async def test_falls_back_to_next_model_on_failure() -> None:
    llm = FakeLlmProvider({"a": RuntimeError("boom"), "b": ["ok"]})
    breaker = FakeCircuitBreaker()
    service = FallbackService(llm, breaker)

    result = await _collect(service, ["a", "b"])

    assert result == [("b", "ok")]
    assert breaker.failures == ["a"]
    assert breaker.successes == ["b"]


async def test_all_models_failing_raises_with_attempted_list() -> None:
    llm = FakeLlmProvider({"a": RuntimeError("x"), "b": RuntimeError("y")})
    breaker = FakeCircuitBreaker()
    service = FallbackService(llm, breaker)

    with pytest.raises(AllFallbacksFailedError) as exc_info:
        await _collect(service, ["a", "b"])

    assert exc_info.value.attempted == ["a", "b"]
