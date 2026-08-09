import re
from typing import Literal

# 다중 주제 비교/분석이 명확히 필요한 경우에만 agentic 경로 사용
# "및", "또는" 같은 일반 접속사는 complex 트리거에서 제외
KO_COMPLEX_KEYWORDS = ["비교해", "비교하", "차이", "차이점", "어떻게 다른", "왜", "원인"]
EN_COMPLEX_PATTERN = re.compile(
    r"\b(compare|difference between|why|explain how|analyze|versus|vs\.)\b", re.IGNORECASE
)
COMPLEX_WORD_THRESHOLD = 10


class QueryComplexityRouter:
    def route(self, question: str) -> Literal["simple", "complex"]:
        words = [w for w in question.strip().split() if w]
        has_ko_keyword = any(kw in question for kw in KO_COMPLEX_KEYWORDS)
        is_long = len(words) >= COMPLEX_WORD_THRESHOLD
        if has_ko_keyword or EN_COMPLEX_PATTERN.search(question) or is_long:
            return "complex"
        return "simple"
