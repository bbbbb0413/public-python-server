from dataclasses import dataclass

from ai_service.llm_gateway.domain.model.llm_message import LlmMessage


@dataclass(frozen=True)
class GatewayCallCommand:
    messages: list[LlmMessage]
    feature: str
    tenant: str
    preferred_model: str | None = None
