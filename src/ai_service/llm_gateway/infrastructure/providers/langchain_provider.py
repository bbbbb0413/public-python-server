from collections.abc import AsyncIterator, Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from ai_service.llm_gateway.domain.model.llm_message import LlmMessage, LlmOptions


def _to_langchain_messages(
    messages: list[LlmMessage],
) -> list[SystemMessage | HumanMessage | AIMessage]:
    converted: list[SystemMessage | HumanMessage | AIMessage] = []
    for m in messages:
        if m.role == "system":
            converted.append(SystemMessage(m.content))
        elif m.role == "assistant":
            converted.append(AIMessage(m.content))
        else:
            converted.append(HumanMessage(m.content))
    return converted


class LangChainLlmProvider:
    """단일 ChatModel 팩토리를 감싸는 ILlmProvider 구현체.

    provider(claude/openai/gemini/groq/ollama)별 개별 클래스를 두지 않고,
    LangChain ChatModel 생성 팩토리만 주입받아 재사용한다.
    """

    def __init__(self, chat_model_factory: Callable[[LlmOptions | None], BaseChatModel]) -> None:
        self._chat_model_factory = chat_model_factory

    async def chat(self, messages: list[LlmMessage], options: LlmOptions | None = None) -> str:
        model = self._chat_model_factory(options)
        response = await model.ainvoke(_to_langchain_messages(messages))
        content = response.content
        return content if isinstance(content, str) else str(content)

    async def stream(
        self, messages: list[LlmMessage], options: LlmOptions | None = None
    ) -> AsyncIterator[str]:
        model = self._chat_model_factory(options)
        async for chunk in model.astream(_to_langchain_messages(messages)):
            if isinstance(chunk.content, str) and chunk.content:
                yield chunk.content
