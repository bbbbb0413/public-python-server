import logging
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any

from ai_service.llm_gateway.schemas import LlmMessage, LlmOptions

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
    def __init__(self, llm: Any, breaker: Any) -> None:
        self._llm = llm
        self._breaker = breaker

    async def stream_with_fallback(
        self,
        messages: list[LlmMessage],
        chain: Sequence[str],
        options: LlmOptions | None = None,
    ) -> AsyncIterator[StreamChunk]:
        attempted: list[str] = []

        for model in chain:
            if not await self._breaker.can_call(model):
                logger.warning("서킷 브레이커 차단으로 모델 스킵: %s", model)
                continue

            attempted.append(model)
            model_options = self._override_model(options, model)

            try:
                async for token in self._llm.stream(messages, model_options):
                    yield StreamChunk(model=model, token=token)
                await self._breaker.record_success(model)
                return
            except Exception as e:  # noqa: BLE001
                logger.error("모델 %s 호출 실패: %s", model, e)
                await self._breaker.record_failure(model)

        raise AllFallbacksFailedError(attempted)

    @staticmethod
    def _override_model(options: LlmOptions | None, model: str) -> LlmOptions:
        if options is None:
            return LlmOptions(model=model)
        return LlmOptions(
            model=model,
            temperature=options.temperature,
            max_tokens=options.max_tokens,
            stream=options.stream,
        )


__all__ = ["AllFallbacksFailedError", "FallbackService", "StreamChunk"]
