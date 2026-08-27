import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

from ai_service.core.config import Settings
from ai_service.observability.repository import RagasEvaluationRepository
from ai_service.observability.schemas import (
    RagasEvalPayload,
    RagasEvaluation,
    RagasScores,
)

logger = logging.getLogger(__name__)

NEUTRAL_CONTEXT_PRECISION = 0.5

LlmEvalFn = Callable[[str], Awaitable[RagasScores]]


class RagasEvalService:
    """RAGAS 스타일 답변 품질 평가.

    LLM 평가(``llm_eval``)가 주입되지 않으면 휴리스틱 점수로 폴백한다.
    테스트에서는 ``llm_eval``에 가짜 콜러블을 주입해 LLM 호출 없이 검증한다.
    """

    def __init__(
        self,
        repo: RagasEvaluationRepository | Any,
        settings: Settings,
        llm_eval: LlmEvalFn | None = None,
    ) -> None:
        self._repo = repo
        self._llm_eval = llm_eval
        if llm_eval is None and settings.ragas_llm_eval_enabled and settings.openai_api_key:
            self._llm_eval = self._build_llm_eval(settings.openai_api_key)

    @staticmethod
    def _build_llm_eval(api_key: str) -> LlmEvalFn | None:
        try:
            from langchain_openai import ChatOpenAI
            from pydantic import BaseModel, Field

            class _RagasScoreSchema(BaseModel):
                faithfulness: float = Field(ge=0, le=1)
                answer_relevancy: float = Field(ge=0, le=1)
                context_precision: float = Field(ge=0, le=1)

            model_kwargs: dict[str, Any] = {
                "model": "gpt-4o-mini",
                "temperature": 0,
                "api_key": api_key,
            }
            model = ChatOpenAI(**model_kwargs)
            structured = model.with_structured_output(_RagasScoreSchema)

            async def _invoke(prompt: str) -> RagasScores:
                result = await structured.ainvoke(prompt)
                return RagasScores(
                    faithfulness=result.faithfulness,  # type: ignore[union-attr]
                    answer_relevancy=result.answer_relevancy,  # type: ignore[union-attr]
                    context_precision=result.context_precision,  # type: ignore[union-attr]
                )

            return _invoke
        except Exception as e:  # noqa: BLE001 - 초기화 실패 시 휴리스틱으로 폴백
            logger.warning("RAGAS LLM 평가 초기화 실패 (휴리스틱으로 폴백): %s", e)
            return None

    async def evaluate(self, payload: RagasEvalPayload) -> None:
        if self._llm_eval is not None:
            try:
                scores = await self._score_llm(payload)
            except Exception as e:  # noqa: BLE001 - LLM 평가 실패 시 휴리스틱 폴백
                logger.warning("RAGAS LLM 평가 실패 (휴리스틱 폴백): %s", e)
                scores = self._score_heuristic(payload)
        else:
            scores = self._score_heuristic(payload)

        await self._repo.persist(
            RagasEvaluation(
                trace_id=payload.trace_id,
                question=payload.question,
                faithfulness=scores.faithfulness,
                answer_relevancy=scores.answer_relevancy,
                context_precision=scores.context_precision,
                sampled_at=datetime.now(UTC),
            )
        )

    async def _score_llm(self, payload: RagasEvalPayload) -> RagasScores:
        if self._llm_eval is None:
            raise RuntimeError("llm_eval이 초기화되지 않았습니다.")
        contexts_text = "\n".join(f"[{i + 1}] {c}" for i, c in enumerate(payload.contexts))
        prompt = f"""아래 질문, 답변, 컨텍스트를 평가하여 0~1 사이의 점수를 반환하세요.

질문: {payload.question}
답변: {payload.answer}
컨텍스트:
{contexts_text}

평가 기준:
- faithfulness: 답변이 컨텍스트 사실에 기반하는 정도 (0=전혀 없음, 1=완전히 기반)
- answerRelevancy: 답변이 질문에 관련된 정도 (0=무관, 1=완전히 관련)
- contextPrecision: 컨텍스트가 질문 답변에 적합한 정도 (0=무관, 1=완전히 적합)"""
        return await self._llm_eval(prompt)

    def _score_heuristic(self, payload: RagasEvalPayload) -> RagasScores:
        return RagasScores(
            faithfulness=self._score_faithfulness(payload.answer, payload.contexts),
            answer_relevancy=self._score_answer_relevancy(payload.answer, payload.question),
            context_precision=self._score_context_precision(payload.contexts),
        )

    @staticmethod
    def _score_faithfulness(answer: str, contexts: list[str]) -> float:
        if not answer or not contexts:
            return 0.0
        total_length = sum(len(c) for c in contexts)
        if total_length == 0:
            return 0.0
        words = [w for w in answer.split(" ") if len(w) > 2]
        overlap = sum(1 for c in contexts if any(w in c for w in words))
        return min(overlap / len(contexts), 1.0)

    @staticmethod
    def _score_answer_relevancy(answer: str, question: str) -> float:
        if not answer or not question:
            return 0.0
        question_words = set(question.lower().split())
        answer_words = answer.lower().split()
        matched = sum(1 for w in answer_words if w in question_words)
        return min(matched / max(len(question_words), 1), 1.0)

    @staticmethod
    def _score_context_precision(contexts: list[str]) -> float:
        if not contexts:
            return 0.0
        return NEUTRAL_CONTEXT_PRECISION


__all__ = ["LlmEvalFn", "NEUTRAL_CONTEXT_PRECISION", "RagasEvalService"]
