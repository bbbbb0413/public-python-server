import re

from ai_service.llm_gateway.domain.model.llm_message import LlmMessage
from ai_service.llm_gateway.domain.port.llm_provider_port import ILlmProvider

MAX_SUB_QUERIES = 3
COMPOUND_INDICATORS = [
    "그리고",
    "또한",
    "아울러",
    "뿐만 아니라",
    "더불어",
    "and also",
    "additionally",
    "furthermore",
    "moreover",
]

_LEADING_DASH_PATTERN = re.compile(r"^-\s*")


class QueryDecomposer:
    def __init__(self, llm_provider: ILlmProvider) -> None:
        self._llm_provider = llm_provider

    def should_decompose(self, question: str) -> bool:
        lower = question.lower()
        has_compound = any(indicator in lower for indicator in COMPOUND_INDICATORS)
        multiple_questions = question.count("?") > 1
        return has_compound or multiple_questions

    async def decompose(self, question: str) -> list[str]:
        messages = [
            LlmMessage(
                role="user",
                content=(
                    f"다음 복합 질문을 최대 {MAX_SUB_QUERIES}개의 독립적인 단순 질문으로 "
                    "분해하세요.\n"
                    '각 질문을 새 줄에 "- "로 시작하여 나열하세요. 질문만 출력하세요.\n\n'
                    f"질문: {question}"
                ),
            )
        ]

        tokens = [token async for token in self._llm_provider.stream(messages)]
        raw = "".join(tokens).strip()

        sub_queries = [_LEADING_DASH_PATTERN.sub("", line).strip() for line in raw.split("\n")]
        sub_queries = [line for line in sub_queries if line][:MAX_SUB_QUERIES]

        return sub_queries if sub_queries else [question]
