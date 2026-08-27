import re
from typing import Any

from ai_service.llm_gateway.schemas import LlmMessage

MIN_QUERY_WORDS = 3
MAX_QUERY_WORDS = 10

_WHITESPACE_PATTERN = re.compile(r"\s+")


class HydeService:
    def __init__(self, llm_provider: Any) -> None:
        self._llm_provider = llm_provider

    def _get_word_count(self, text: str) -> int:
        space_words = _WHITESPACE_PATTERN.split(text.strip())
        if len(space_words) == 1:
            return -(-len(re.sub(r"\s", "", text)) // 3)
        return len(space_words)

    def should_apply(self, question: str) -> bool:
        trimmed = question.strip()
        if not trimmed:
            return False
        word_count = self._get_word_count(trimmed)
        return MIN_QUERY_WORDS <= word_count <= MAX_QUERY_WORDS

    async def generate_hypothetical_document(self, question: str) -> str | None:
        if not self.should_apply(question):
            return None
        return await self.generate_hypothetical(question)

    async def generate_hypothetical(self, question: str) -> str:
        messages = [
            LlmMessage(
                role="system",
                content=(
                    "다음 질문에 대해 간결하고 사실적인 답변을 2-3문장으로 작성하세요. "
                    "이 답변은 문서 검색 쿼리로 활용됩니다."
                ),
            ),
            LlmMessage(role="user", content=question),
        ]

        tokens = [token async for token in self._llm_provider.stream(messages)]
        return "".join(tokens)


__all__ = ["HydeService", "MAX_QUERY_WORDS", "MIN_QUERY_WORDS"]
