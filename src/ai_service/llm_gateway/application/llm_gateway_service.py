import asyncio
import math
import time
from collections.abc import AsyncIterator

from ai_service.llm_gateway.application.cost_tracking_service import (
    CostTrackingService,
    TrackParams,
)
from ai_service.llm_gateway.application.fallback_service import FallbackService
from ai_service.llm_gateway.application.langsmith_tracing_service import (
    LangSmithTracingService,
    LlmRunParams,
)
from ai_service.llm_gateway.application.llm_routing_service import LlmRoutingService
from ai_service.llm_gateway.schemas import GatewayCallCommand, TokenUsage

APPROX_CHARS_PER_TOKEN = 4


class LlmGatewayService:
    def __init__(
        self,
        fallback: FallbackService,
        cost_tracking: CostTrackingService,
        routing: LlmRoutingService,
        langsmith_tracing: LangSmithTracingService | None = None,
    ) -> None:
        self._fallback = fallback
        self._cost_tracking = cost_tracking
        self._routing = routing
        self._langsmith_tracing = langsmith_tracing

    async def stream(self, command: GatewayCallCommand) -> AsyncIterator[str]:
        chain = self._routing.resolve_chain(command.preferred_model)
        start_time = time.time()

        used_model = chain[0]
        prompt_tokens = sum(
            math.ceil(len(m.content) / APPROX_CHARS_PER_TOKEN) for m in command.messages
        )
        completion_tokens = 0
        attempted_models: set[str] = set()
        output_tokens: list[str] = []

        async for chunk in self._fallback.stream_with_fallback(command.messages, chain):
            used_model = chunk.model
            attempted_models.add(chunk.model)
            if chunk.token is not None:
                completion_tokens += 1
                output_tokens.append(chunk.token)
                yield chunk.token

        await self._cost_tracking.track(
            TrackParams(
                model=used_model,
                feature=command.feature,
                tenant=command.tenant,
                usage=TokenUsage.of(prompt_tokens, completion_tokens),
                fallback_used=used_model != chain[0],
                attempted_models=list(attempted_models),
            )
        )

        if self._langsmith_tracing is not None:
            asyncio.create_task(  # noqa: RUF006 - 완료를 기다리지 않는 fire-and-forget 트레이싱
                self._langsmith_tracing.log_llm_run(
                    LlmRunParams(
                        name="llm-gateway",
                        messages=[{"role": m.role, "content": m.content} for m in command.messages],
                        answer="".join(output_tokens),
                        model=used_model,
                        completion_tokens=completion_tokens,
                        feature=command.feature,
                        tenant=command.tenant,
                        start_time=start_time,
                        end_time=time.time(),
                    )
                )
            )
