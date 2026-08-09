import logging
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass

from ai_service.llm_gateway.domain.model.llm_message import LlmMessage, LlmOptions
from ai_service.llm_gateway.domain.port.circuit_breaker_port import ICircuitBreakerPort
from ai_service.llm_gateway.domain.port.llm_provider_port import ILlmProvider

logger = logging.getLogger(__name__)


class AllFallbacksFailedError(Exception):
    def __init__(self, attempted: list[str]) -> None:
        super().__init__(f"모든 폴백 실패. 시도: {' → '.join(attempted)}")
        self.attempted = attempted


@dataclass(frozen=True)
class StreamChunk:
    model: str
    token: str | None = None


class FallbackService:
    def __init__(self, llm: ILlmProvider, breaker: ICircuitBreakerPort) -> None:
        self._llm = llm
        self._breaker = breaker

    async def stream_with_fallback(
        self, messages: list[LlmMessage], chain: Sequence[str]
    ) -> AsyncIterator[StreamChunk]:
        attempted: list[str] = []

        for model in chain:
            attempted.append(model)

            if not await self._breaker.can_call(model):
                logger.warning("회로 개방으로 모델 건너뜀: %s", model)
                continue

            try:
                async for token in self._llm.stream(messages, LlmOptions(model=model)):
                    yield StreamChunk(model=model, token=token)
                await self._breaker.record_success(model)
                return
            except Exception as e:  # noqa: BLE001 - 폴백 대상 오류를 모두 포괄해야 함
                logger.error("모델 호출 실패(%s) → 폴백: %s", model, e)
                await self._breaker.record_failure(model)

        raise AllFallbacksFailedError(attempted)
