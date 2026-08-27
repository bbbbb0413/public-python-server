import json
import logging
import re
from typing import Any, Protocol

from pydantic import BaseModel, Field, SecretStr

from ai_service.core.config import Settings
from ai_service.knowledge.schemas import SimilaritySearchResult
from ai_service.llm_gateway.application.llm_gateway_service import LlmGatewayService
from ai_service.llm_gateway.schemas import GatewayCallCommand, LlmMessage
from ai_service.rag.schemas import Critique

logger = logging.getLogger(__name__)

CRITIQUE_PROMPT_NAME = "rag-critique"
DEFAULT_TENANT = "default"


class CritiqueOutput(BaseModel):
    answered: bool
    missing: list[str]
    next_query: str = Field(alias="nextQuery")
    confidence: float


class IStructuredModel(Protocol):
    async def ainvoke(self, prompt: str) -> CritiqueOutput: ...


class CritiqueGeneratorService:
    def __init__(self, llm_gateway: LlmGatewayService, settings: Settings) -> None:
        self._llm_gateway = llm_gateway
        self.structured_model: IStructuredModel | None = None

        if settings.openai_api_key:
            self.structured_model = self._init_structured_model(settings.openai_api_key)

    @staticmethod
    def _init_structured_model(api_key: str) -> IStructuredModel | None:
        try:
            from langchain_openai import ChatOpenAI

            model = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=SecretStr(api_key))
            return model.with_structured_output(CritiqueOutput)  # type: ignore[return-value]
        except Exception as e:
            logger.warning("CritiqueGenerator structuredOutput 초기화 실패: %s", e)
            return None

    async def generate(
        self,
        question: str,
        answer: str,
        chunks: list[SimilaritySearchResult],
        tenant: str | None = None,
    ) -> Critique:
        if self.structured_model is not None:
            try:
                return await self._generate_with_structured_output(question, answer, chunks)
            except Exception as e:
                logger.warning("structuredOutput 실패 (스트리밍 폴백): %s", e)

        return await self._generate_with_stream(question, answer, chunks, tenant)

    async def _generate_with_structured_output(
        self, question: str, answer: str, chunks: list[SimilaritySearchResult]
    ) -> Critique:
        context_summary = self._build_context_summary(chunks)
        prompt = "\n".join(
            [
                "아래 질문, 검색된 컨텍스트, 생성된 답변을 평가하세요.",
                "",
                f"질문: {question}",
                f"컨텍스트:\n{context_summary}",
                f"답변: {answer}",
            ]
        )

        assert self.structured_model is not None
        result = await self.structured_model.ainvoke(prompt)
        return Critique.of(result.answered, result.missing, result.next_query, result.confidence)

    async def _generate_with_stream(
        self,
        question: str,
        answer: str,
        chunks: list[SimilaritySearchResult],
        tenant: str | None,
    ) -> Critique:
        context_summary = self._build_context_summary(chunks)

        system_prompt = "\n".join(
            [
                "당신은 RAG 시스템의 답변 품질을 평가하는 전문가입니다.",
                "아래 질문, 검색된 컨텍스트, 생성된 답변을 보고 JSON으로만 응답하세요.",
                '{"answered": bool, "missing": [string], "nextQuery": string, '
                '"confidence": float(0-1)}',
            ]
        )
        user_prompt = "\n\n".join(
            [f"질문: {question}", f"컨텍스트:\n{context_summary}", f"답변: {answer}"]
        )

        messages = [
            LlmMessage(role="system", content=system_prompt),
            LlmMessage(role="user", content=user_prompt),
        ]

        collected = [
            token
            async for token in self._llm_gateway.stream(
                GatewayCallCommand(messages, CRITIQUE_PROMPT_NAME, tenant or DEFAULT_TENANT)
            )
        ]

        return self._parse_critique("".join(collected))

    @staticmethod
    def _build_context_summary(chunks: list[SimilaritySearchResult]) -> str:
        return "\n".join(f"[{i + 1}] {c.text[:200]}" for i, c in enumerate(chunks[:3]))

    def _parse_critique(self, raw: str) -> Critique:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return self._fallback_critique()

        try:
            parsed: dict[str, Any] = json.loads(match.group(0))
        except json.JSONDecodeError:
            return self._fallback_critique()

        answered = parsed.get("answered") is True
        missing_raw = parsed.get("missing")
        missing = (
            [m for m in missing_raw if isinstance(m, str)] if isinstance(missing_raw, list) else []
        )
        next_query_raw = parsed.get("nextQuery")
        next_query = next_query_raw if isinstance(next_query_raw, str) else ""
        confidence_raw = parsed.get("confidence")
        confidence = (
            confidence_raw
            if isinstance(confidence_raw, int | float) and 0 <= confidence_raw <= 1
            else 0.7
        )

        return Critique.of(answered, missing, next_query, float(confidence))

    @staticmethod
    def _fallback_critique() -> Critique:
        # 파싱 실패 시 첫 번째 답변을 그대로 사용 (임계값 0.6 이상으로 재반복 방지)
        return Critique.of(True, [], "", 0.7)
