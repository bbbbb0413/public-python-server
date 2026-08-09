from ai_service.config.settings import Settings
from ai_service.llm_gateway.application.command.gateway_call_command import (
    GatewayCallCommand,
)
from ai_service.llm_gateway.application.cost_tracking_service import (
    CostTrackingService,
    ModelCostEntry,
)
from ai_service.llm_gateway.application.fallback_service import FallbackService
from ai_service.llm_gateway.application.llm_gateway_service import LlmGatewayService
from ai_service.llm_gateway.application.llm_routing_service import LlmRoutingService
from ai_service.llm_gateway.domain.model.llm_message import LlmMessage
from tests.unit.llm_gateway.fakes import (
    FakeCircuitBreaker,
    FakeLlmCostLogRepository,
    FakeLlmProvider,
)


async def _collect_stream(service: LlmGatewayService, command: GatewayCallCommand) -> str:
    tokens = [token async for token in service.stream(command)]
    return "".join(tokens)


async def test_stream_yields_tokens_and_tracks_cost() -> None:
    llm = FakeLlmProvider({"primary": ["he", "llo"]})
    breaker = FakeCircuitBreaker()
    fallback = FallbackService(llm, breaker)
    cost_repo = FakeLlmCostLogRepository()
    cost_tracking = CostTrackingService(
        cost_repo, {"primary": ModelCostEntry(prompt=1.0, completion=1.0)}
    )
    routing = LlmRoutingService(Settings(llm_provider="ollama", ollama_model="primary"))
    service = LlmGatewayService(fallback, cost_tracking, routing)

    command = GatewayCallCommand(
        messages=[LlmMessage(role="user", content="hi")], feature="qa", tenant="default"
    )
    answer = await _collect_stream(service, command)

    assert answer == "hello"
    assert len(cost_repo.logs) == 1
    assert cost_repo.logs[0].model == "primary"
    assert cost_repo.logs[0].fallback_used is False


async def test_stream_marks_fallback_used_when_secondary_model_wins() -> None:
    llm = FakeLlmProvider({"primary": RuntimeError("boom"), "secondary": ["ok"]})
    breaker = FakeCircuitBreaker()
    fallback = FallbackService(llm, breaker)
    cost_repo = FakeLlmCostLogRepository()
    cost_tracking = CostTrackingService(cost_repo, {})
    routing = LlmRoutingService(
        Settings(llm_provider="ollama", ollama_model="primary", llm_fallback_chain="secondary")
    )
    service = LlmGatewayService(fallback, cost_tracking, routing)

    command = GatewayCallCommand(
        messages=[LlmMessage(role="user", content="hi")], feature="qa", tenant="default"
    )
    answer = await _collect_stream(service, command)

    assert answer == "ok"
    assert cost_repo.logs[0].fallback_used is True
    assert cost_repo.logs[0].model == "secondary"
