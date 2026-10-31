from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import pytest

from ai_service.knowledge.schemas import (
    SimilaritySearchResult,
    VectorDocumentMetadata,
)
from ai_service.llm_gateway.application.llm_gateway_service import LlmGatewayService
from ai_service.prompt.schemas import PromptTemplate
from ai_service.prompt.service import PromptService
from ai_service.rag.application.agentic_ask_use_case import AgenticAskUseCase
from ai_service.rag.application.command.agentic_ask_command import AgenticAskCommand
from ai_service.rag.application.critique_generator_service import CritiqueGeneratorService
from ai_service.rag.application.filter.rag_content_validator import RagContentValidator
from ai_service.rag.application.filter.secret_pii_scanner import SecretPiiScanner
from ai_service.rag.application.hybrid_search_use_case import (
    HybridSearchResult,
    HybridSearchUseCase,
)
from ai_service.rag.application.query_refiner_service import QueryRefinerService
from ai_service.rag.schemas import Critique, IterationBudget


def _make_prompt_template(name: str = "rag-qa-system") -> PromptTemplate:
    return PromptTemplate.create(
        name,
        "Context: {{context}}\nDate: {{currentDate}}",
        ["context", "currentDate"],
    )


@pytest.fixture
def mock_hybrid_search() -> AsyncMock:
    mock = AsyncMock(spec=HybridSearchUseCase)
    chunk = SimilaritySearchResult(
        text="문서 본문",
        score=0.85,
        metadata=VectorDocumentMetadata(
            file_name="test.txt",
            chunk_index=0,
            document_id="doc-1",
        ),
    )
    mock.execute.return_value = HybridSearchResult(chunks=[chunk], query_embedding=[0.1, 0.2])
    return mock


@pytest.fixture
def mock_llm_gateway() -> MagicMock:
    mock = MagicMock(spec=LlmGatewayService)

    async def fake_stream(_command) -> AsyncIterator[str]:
        for token in ["테스트", " ", "답변입니다."]:
            yield token

    mock.stream.side_effect = fake_stream
    return mock


@pytest.fixture
def mock_get_active_prompt() -> AsyncMock:
    mock = AsyncMock(spec=PromptService)
    mock.execute.return_value = _make_prompt_template()
    return mock


@pytest.fixture
def mock_critique_generator() -> AsyncMock:
    return AsyncMock(spec=CritiqueGeneratorService)


@pytest.fixture
def mock_query_refiner() -> MagicMock:
    return MagicMock(spec=QueryRefinerService)


@pytest.fixture
def rag_validator() -> RagContentValidator:
    return RagContentValidator()


@pytest.fixture
def secret_pii_scanner() -> SecretPiiScanner:
    return SecretPiiScanner()


@pytest.mark.asyncio
async def test_agentic_ask_emits_next_query_in_refining_and_next_searching(
    mock_hybrid_search: AsyncMock,
    mock_llm_gateway: MagicMock,
    mock_get_active_prompt: AsyncMock,
    mock_critique_generator: AsyncMock,
    mock_query_refiner: MagicMock,
    rag_validator: RagContentValidator,
    secret_pii_scanner: SecretPiiScanner,
):
    # 1회차 불만족 -> 2회차 만족
    mock_critique_generator.generate.side_effect = [
        Critique.of(
            answered=False,
            missing=["환불 세부 절차"],
            next_query="환불 세부 절차",
            confidence=0.4,
        ),
        Critique.of(answered=True, missing=[], next_query="", confidence=0.92),
    ]
    mock_query_refiner.refine.return_value = "정제된 환불 질의"

    events: list[dict] = []

    async def on_progress(data: dict) -> None:
        events.append(data)

    use_case = AgenticAskUseCase(
        hybrid_search=mock_hybrid_search,
        llm_gateway=mock_llm_gateway,
        get_active_prompt=mock_get_active_prompt,
        critique_generator=mock_critique_generator,
        query_refiner=mock_query_refiner,
        rag_validator=rag_validator,
        secret_pii_scanner=secret_pii_scanner,
    )

    command = AgenticAskCommand(
        question="환불 절차 안내",
        budget=IterationBudget.of(max_iterations=3, token_budget=1000, timeout_ms=5000),
        confidence_threshold=0.8,
        on_progress=on_progress,
    )

    chunks = [c async for c in use_case.execute(command)]
    assert len(chunks) > 0

    # 1회차: searching(0) -> generating(1) -> critiquing(2) -> refining(3)
    # 2회차: searching(4) -> generating(5) -> critiquing(6) -> 완료 critiquing(7)
    assert len(events) == 8

    # 1회차 refining 단계에서 nextQuery 전달 확인
    refining_event = events[3]
    assert refining_event["phase"] == "refining"
    assert refining_event.get("nextQuery") == "정제된 환불 질의"

    # 2회차 searching 단계에서 nextQuery 전달 확인
    searching_event = events[4]
    assert searching_event["phase"] == "searching"
    assert searching_event["iteration"] == 2
    assert searching_event.get("nextQuery") == "정제된 환불 질의"

    # 2회차 generating, critiquing 단계에서는 nextQuery가 없거나 None
    assert events[5]["phase"] == "generating"
    assert events[5].get("nextQuery") is None

    # 최종 완료 이벤트(events[7])에 누적된 refinedQueries 포함 확인
    done_event = events[7]
    assert done_event["phase"] == "critiquing"
    assert done_event.get("refinedQueries") == ["정제된 환불 질의"]


@pytest.mark.asyncio
async def test_agentic_ask_multi_refinements_accumulates_refined_queries(
    mock_hybrid_search: AsyncMock,
    mock_llm_gateway: MagicMock,
    mock_get_active_prompt: AsyncMock,
    mock_critique_generator: AsyncMock,
    mock_query_refiner: MagicMock,
    rag_validator: RagContentValidator,
    secret_pii_scanner: SecretPiiScanner,
):
    # 1회차 불만족 -> 2회차 불만족 -> 3회차 만족
    mock_critique_generator.generate.side_effect = [
        Critique.of(answered=False, missing=["항목1"], next_query="q1", confidence=0.3),
        Critique.of(answered=False, missing=["항목2"], next_query="q2", confidence=0.6),
        Critique.of(answered=True, missing=[], next_query="", confidence=0.95),
    ]
    mock_query_refiner.refine.side_effect = ["정제쿼리1", "정제쿼리2"]

    events: list[dict] = []

    async def on_progress(data: dict) -> None:
        events.append(data)

    use_case = AgenticAskUseCase(
        hybrid_search=mock_hybrid_search,
        llm_gateway=mock_llm_gateway,
        get_active_prompt=mock_get_active_prompt,
        critique_generator=mock_critique_generator,
        query_refiner=mock_query_refiner,
        rag_validator=rag_validator,
        secret_pii_scanner=secret_pii_scanner,
    )

    command = AgenticAskCommand(
        question="복합 질문",
        budget=IterationBudget.of(max_iterations=4, token_budget=2000, timeout_ms=5000),
        confidence_threshold=0.8,
        on_progress=on_progress,
    )

    [c async for c in use_case.execute(command)]

    # 최종 이벤트에서 누적된 쿼리 목록 확인
    final_event = events[-1]
    assert final_event.get("refinedQueries") == ["정제쿼리1", "정제쿼리2"]
