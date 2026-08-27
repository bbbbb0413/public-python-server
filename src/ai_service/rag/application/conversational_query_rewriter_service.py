from typing import Any

from ai_service.llm_gateway.schemas import LlmMessage

MAX_HISTORY_TURNS = 4

FOLLOW_UP_INDICATORS = [
    "이것",
    "그것",
    "저것",
    "위에서",
    "앞에서",
    "해당",
    "이 내용",
    "그 내용",
    "이거",
    "그거",
    "앞서",
    "거기",
    "여기",
    "더",
    "추가로",
    "그러면",
    "그럼",
    "방금",
    "위",
    "아래",
    " it ",
    " this ",
    " that ",
    " those ",
    " they ",
    " there ",
]


class ConversationalQueryRewriter:
    def __init__(self, llm_provider: Any) -> None:
        self._llm_provider = llm_provider

    def is_follow_up(self, question: str, history: list[dict[str, str]]) -> bool:
        if not history:
            return False
        lower = question.lower()
        return any(indicator in lower for indicator in FOLLOW_UP_INDICATORS)

    async def rewrite(self, question: str, history: list[dict[str, str]]) -> str:
        history_text = "\n".join(
            f"{'사용자' if turn['role'] == 'user' else '어시스턴트'}: {turn['content']}"
            for turn in history[-MAX_HISTORY_TURNS:]
        )

        messages = [
            LlmMessage(
                role="user",
                content=(
                    f"이전 대화:\n{history_text}\n\n"
                    "위 대화 맥락을 참고하여, 다음 질문을 이전 대화 없이도 이해할 수 있는 "
                    "독립적인 질문으로 재작성하세요.\n재작성된 질문만 출력하세요.\n\n"
                    f"질문: {question}"
                ),
            )
        ]

        tokens = [token async for token in self._llm_provider.stream(messages)]
        rewritten = "".join(tokens).strip()
        return rewritten or question
