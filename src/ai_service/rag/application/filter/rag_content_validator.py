import re
import unicodedata

from ai_service.knowledge.domain.port.vector_store_port import SimilaritySearchResult
from ai_service.rag.domain.policy.injection_patterns import INJECTION_PATTERNS
from ai_service.rag.domain.vo.guardrail_verdict import GuardrailVerdict

_ZERO_WIDTH_PATTERN = re.compile(r"[​-‍﻿]")


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
                    text="\n".join(safe_lines), score=chunk.score, metadata=chunk.metadata
                )
            )
        return sanitized

    def scan(self, raw_text: str) -> GuardrailVerdict:
        return self.inspect_input(raw_text)

    @staticmethod
    def _clean_text(text: str) -> str:
        return _ZERO_WIDTH_PATTERN.sub("", unicodedata.normalize("NFKC", text))
