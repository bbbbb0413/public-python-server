from collections.abc import AsyncIterator
from typing import Protocol

from ai_service.llm_gateway.domain.model.llm_message import LlmMessage, LlmOptions


class ILlmProvider(Protocol):
    async def chat(self, messages: list[LlmMessage], options: LlmOptions | None = None) -> str: ...

    def stream(
        self, messages: list[LlmMessage], options: LlmOptions | None = None
    ) -> AsyncIterator[str]: ...
