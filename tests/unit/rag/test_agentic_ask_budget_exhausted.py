import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from ai_service.knowledge.schemas import (
    SimilaritySearchResult,
    VectorDocumentMetadata,
)
from ai_service.llm_gateway.application.llm_gateway_service import LlmGatewayService
from ai_service.prompt.service import PromptService
from ai_service.rag.application.agentic_ask_use_case import (
    AgenticAskCommand,
    AgenticAskUseCase,
)
from ai_service.rag.application.critique_generator_service import (
    Critique,
    CritiqueGeneratorService,
)
from ai_service.rag.application.filter.rag_content_validator import RagContentValidator
from ai_service.rag.application.filter.secret_pii_scanner import SecretPiiScanner
from ai_service.rag.application.hybrid_search_use_case import (
    HybridSearchResult,
    HybridSearchUseCase,
)
from ai_service.rag.application.query_refiner_service import QueryRefinerService
from ai_service.rag.schemas import (
    IterationBudget,
)


@pytest.mark.asyncio
async def test_agentic_ask_first_iteration_exhausted_emits_budget_exhausted_metadata():
    # 1회차에서 청크 점수가 0.95로 높더라도 예산 소진 시:
    # 1. confidence=0.0, missing=[]
    # 2. progress 이벤트에 isBudgetExhausted=True 포함
    # 3. 스트림 청크로 __CONFIDENCE 메타데이터(confidence: 0.0, isBudgetExhausted: True) 방출
    chunks = [
        SimilaritySearchResult(
            text="청크 내용 1",
            score=0.95,
            metadata=VectorDocumentMetadata(document_id="doc-1", file_name="f1.pdf", chunk_index=0),
        )
    ]
    mock_hybrid_search = AsyncMock(spec=HybridSearchUseCase)
    mock_hybrid_search.execute.return_value = HybridSearchResult(
        chunks=chunks, query_embedding=[0.1, 0.2]
    )

    mock_llm_gateway = MagicMock(spec=LlmGatewayService)

    async def fake_stream(_command):
        yield "답변 내용입니다."

    mock_llm_gateway.stream.side_effect = fake_stream

    mock_prompt_service = AsyncMock(spec=PromptService)
    from ai_service.prompt.schemas import PromptTemplate

    mock_prompt_service.execute.return_value = PromptTemplate.create(
        "rag-qa-system", "Context: {{context}}", ["context"]
    )

    mock_critique = AsyncMock(spec=CritiqueGeneratorService)
    mock_refiner = MagicMock(spec=QueryRefinerService)

    events: list[dict] = []

    async def on_progress(data: dict) -> None:
        events.append(data)

    use_case = AgenticAskUseCase(
        hybrid_search=mock_hybrid_search,
        llm_gateway=mock_llm_gateway,
        get_active_prompt=mock_prompt_service,
        critique_generator=mock_critique,
        query_refiner=mock_refiner,
        rag_validator=RagContentValidator(),
        secret_pii_scanner=SecretPiiScanner(),
        on_progress=on_progress,
    )

    command = AgenticAskCommand(
        question="1회차 예산 소진 테스트",
        budget=IterationBudget.of(max_iterations=1, token_budget=1, timeout_ms=5000),
        confidence_threshold=0.8,
    )

    stream_chunks = [c async for c in use_case.execute(command)]

    # 비평은 실행되지 않아야 함
    mock_critique.generate.assert_not_awaited()

    # 스트림 청크 마지막에 __CONFIDENCE 메타데이터가 방출되었는지 확인
    confidence_chunks = [c for c in stream_chunks if c.startswith("__CONFIDENCE:")]
    assert len(confidence_chunks) == 1
    meta = json.loads(confidence_chunks[0][len("__CONFIDENCE:") :])
    assert meta == {
        "confidence": 0.0,
        "missing": [],
        "isBudgetExhausted": True,
    }

    # 마지막 progress 이벤트 확인
    final_event = events[-1]
    assert final_event["iteration"] == 1
    assert final_event["phase"] == "generating"
    assert final_event["confidence"] == 0.0
    assert final_event["missing"] == []
    assert final_event["isBudgetExhausted"] is True


@pytest.mark.asyncio
async def test_agentic_ask_subsequent_iteration_exhausted_normalizes_to_zero_and_emits_metadata():
    # 2회차에서 예산 소진 시 1회차 비평 점수가 0.7이었더라도:
    # 1. confidence=0.0, missing=[]
    # 2. progress 이벤트에 isBudgetExhausted=True 포함
    # 3. 스트림 청크로 __CONFIDENCE 메타데이터 방출
    mock_hybrid_search = AsyncMock(spec=HybridSearchUseCase)
    mock_hybrid_search.execute.return_value = HybridSearchResult(
        chunks=[
            SimilaritySearchResult(
                text="청크 내용",
                score=0.88,
                metadata=VectorDocumentMetadata(
                    document_id="doc-1", file_name="f1.pdf", chunk_index=0
                ),
            )
        ],
        query_embedding=[0.1, 0.2],
    )

    mock_llm_gateway = MagicMock(spec=LlmGatewayService)

    async def fake_stream(_command):
        yield "답변 내용입니다."

    mock_llm_gateway.stream.side_effect = fake_stream

    mock_prompt_service = AsyncMock(spec=PromptService)
    from ai_service.prompt.schemas import PromptTemplate

    mock_prompt_service.execute.return_value = PromptTemplate.create(
        "rag-qa-system", "Context: {{context}}", ["context"]
    )

    mock_critique = AsyncMock(spec=CritiqueGeneratorService)
    mock_critique.generate.return_value = Critique.of(
        answered=False,
        missing=["추가 보완"],
        next_query="보완 검색",
        confidence=0.7,
    )

    mock_refiner = MagicMock(spec=QueryRefinerService)
    mock_refiner.refine.return_value = "보완 검색 쿼리"

    events: list[dict] = []

    async def on_progress(data: dict) -> None:
        events.append(data)

    use_case = AgenticAskUseCase(
        hybrid_search=mock_hybrid_search,
        llm_gateway=mock_llm_gateway,
        get_active_prompt=mock_prompt_service,
        critique_generator=mock_critique,
        query_refiner=mock_refiner,
        rag_validator=RagContentValidator(),
        secret_pii_scanner=SecretPiiScanner(),
        on_progress=on_progress,
    )

    command = AgenticAskCommand(
        question="2회차 예산 소진 테스트",
        budget=IterationBudget.of(max_iterations=2, token_budget=1000, timeout_ms=5000),
        confidence_threshold=0.8,
    )

    stream_chunks = [c async for c in use_case.execute(command)]

    confidence_chunks = [c for c in stream_chunks if c.startswith("__CONFIDENCE:")]
    assert len(confidence_chunks) == 1
    meta = json.loads(confidence_chunks[0][len("__CONFIDENCE:") :])
    assert meta == {
        "confidence": 0.0,
        "missing": [],
        "isBudgetExhausted": True,
    }

    final_event = events[-1]
    assert final_event["iteration"] == 2
    assert final_event["phase"] == "generating"
    assert final_event["confidence"] == 0.0
    assert final_event["missing"] == []
    assert final_event["isBudgetExhausted"] is True
