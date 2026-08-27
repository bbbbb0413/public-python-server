import re
import unicodedata

from ai_service.knowledge.schemas import SimilaritySearchResult
from ai_service.rag.schemas import GuardrailVerdict

_ZERO_WIDTH_PATTERN = re.compile(r"[\u200b-\u200d\ufeff]")

INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(previous|all)\s+instructions?", re.IGNORECASE),
    re.compile(r"system\s*:\s*", re.IGNORECASE),
    re.compile(r"당신의\s*이전\s*지시(사항)?을?\s*무시(해|하세요|하십시오)?", re.IGNORECASE),
    re.compile(r"시스템\s*프롬프트를?\s*(출력|알려줘|보여줘)", re.IGNORECASE),
    re.compile(r"<\s*script[^>]*>", re.IGNORECASE),
]


class RagContentValidator:
    def inspect_input(self, text: str) -> GuardrailVerdict:
        cleaned = self._clean_text(text)
        matched = next((p for p in INJECTION_PATTERNS if p.search(cleaned)), None)
        if matched is None:
            return GuardrailVerdict.allow()
        return GuardrailVerdict.block("의심스러운 지시문 패턴", matched.pattern)

    def sanitize(self, chunks: list[SimilaritySearchResult]) -> list[SimilaritySearchResult]:
        sanitized: list[SimilaritySearchResult] = []
        for chunk in chunks:
            cleaned_lines = self._clean_text(chunk.text).split("\n")
            safe_lines = [
                line
                for line in cleaned_lines
                if not any(p.search(line) for p in INJECTION_PATTERNS)
            ]
            sanitized.append(
                SimilaritySearchResult(
                    text="\n".join(safe_lines),
                    score=chunk.score,
                    metadata=chunk.metadata,
                )
            )
        return sanitized

    @staticmethod
    def _clean_text(text: str) -> str:
        normalized = unicodedata.normalize("NFKC", text)
        return _ZERO_WIDTH_PATTERN.sub("", normalized)


__all__ = ["INJECTION_PATTERNS", "RagContentValidator"]
