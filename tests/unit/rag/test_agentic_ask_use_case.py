from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import pytest

from ai_service.llm_gateway.application.llm_gateway_service import LlmGatewayService
from ai_service.prompt.application.get_active_prompt_use_case import GetActivePromptUseCase
from ai_service.prompt.domain.model.prompt_template import PromptTemplate
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
from ai_service.rag.domain.vo.critique import Critique
from ai_service.rag.domain.vo.iteration_budget import IterationBudget


def _make_prompt_template(name: str = "rag-qa-system") -> PromptTemplate:
    return PromptTemplate.create(
        name,
        "Context: {{context}}\nDate: {{currentDate}}",
        ["context", "currentDate"],
    )


@pytest.fixture
def mock_hybrid_search() -> AsyncMock:
    mock = AsyncMock(spec=HybridSearchUseCase)
    mock.execute.return_value = HybridSearchResult(chunks=[], query_embedding=[0.1, 0.2])
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
    mock = AsyncMock(spec=GetActivePromptUseCase)
    mock.execute.return_value = _make_prompt_template()
    return mock


@pytest.fixture
def mock_critique_generator() -> AsyncMock:
    mock = AsyncMock(spec=CritiqueGeneratorService)
    return mock


@pytest.fixture
def mock_query_refiner() -> MagicMock:
    mock = MagicMock(spec=QueryRefinerService)
    mock.refine.return_value = "개선된 질문"
    return mock


@pytest.fixture
def rag_validator() -> RagContentValidator:
    return RagContentValidator()


@pytest.fixture
def secret_pii_scanner() -> SecretPiiScanner:
    return SecretPiiScanner()


@pytest.mark.asyncio
async def test_agentic_ask_single_iteration_progress_events(
    mock_hybrid_search: AsyncMock,
    mock_llm_gateway: MagicMock,
    mock_get_active_prompt: AsyncMock,
    mock_critique_generator: AsyncMock,
    mock_query_refiner: MagicMock,
    rag_validator: RagContentValidator,
    secret_pii_scanner: SecretPiiScanner,
):
    mock_critique_generator.generate.return_value = Critique.of(
        answered=True, missing=[], next_query="", confidence=0.9
    )

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
        on_progress=on_progress,
    )

    command = AgenticAskCommand(
        question="환불 정책이 어떻게 되나요?",
        budget=IterationBudget.of(max_iterations=3, token_budget=1000, timeout_ms=5000),
        confidence_threshold=0.8,
    )

    chunks = [c async for c in use_case.execute(command)]
    assert "".join(chunks) == "테스트 답변입니다."

    # 1회 반복에서 searching -> generating -> critiquing -> 완료 후 최종 이벤트
    assert len(events) == 4
    assert events[0] == {"iteration": 1, "phase": "searching", "confidence": 0.0, "missing": []}
    assert events[1] == {"iteration": 1, "phase": "generating", "confidence": 0.0, "missing": []}
    assert events[2] == {"iteration": 1, "phase": "critiquing", "confidence": 0.0, "missing": []}
    assert events[3] == {"iteration": 1, "phase": "critiquing", "confidence": 0.9, "missing": []}


@pytest.mark.asyncio
async def test_agentic_ask_multi_iteration_progress_events(
    mock_hybrid_search: AsyncMock,
    mock_llm_gateway: MagicMock,
    mock_get_active_prompt: AsyncMock,
    mock_critique_generator: AsyncMock,
    mock_query_refiner: MagicMock,
    rag_validator: RagContentValidator,
    secret_pii_scanner: SecretPiiScanner,
):
    # 1회차: 만족 못함 (missing: ["결제 취소 정책"], confidence: 0.5)
    # 2회차: 만족함 (missing: [], confidence: 0.95)
    mock_critique_generator.generate.side_effect = [
        Critique.of(
            answered=False,
            missing=["결제 취소 정책"],
            next_query="결제 취소 정책",
            confidence=0.5,
        ),
        Critique.of(answered=True, missing=[], next_query="", confidence=0.95),
    ]

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
        question="결제 및 환불 안내",
        budget=IterationBudget.of(max_iterations=3, token_budget=1000, timeout_ms=5000),
        confidence_threshold=0.8,
        on_progress=on_progress,
    )

    chunks = [c async for c in use_case.execute(command)]
    assert "".join(chunks) == "테스트 답변입니다."

    # 1회차: searching -> generating -> critiquing -> refining
    # 2회차: searching -> generating -> critiquing -> 최종 완료
    assert len(events) == 8
    assert events[0] == {"iteration": 1, "phase": "searching", "confidence": 0.0, "missing": []}
    assert events[1] == {"iteration": 1, "phase": "generating", "confidence": 0.0, "missing": []}
    assert events[2] == {"iteration": 1, "phase": "critiquing", "confidence": 0.0, "missing": []}
    assert events[3] == {
        "iteration": 1,
        "phase": "refining",
        "confidence": 0.5,
        "missing": ["결제 취소 정책"],
    }
    assert events[4] == {
        "iteration": 2,
        "phase": "searching",
        "confidence": 0.5,
        "missing": ["결제 취소 정책"],
    }
    assert events[5] == {
        "iteration": 2,
        "phase": "generating",
        "confidence": 0.5,
        "missing": ["결제 취소 정책"],
    }
    assert events[6] == {
        "iteration": 2,
        "phase": "critiquing",
        "confidence": 0.5,
        "missing": ["결제 취소 정책"],
    }
    assert events[7] == {"iteration": 2, "phase": "critiquing", "confidence": 0.95, "missing": []}


@pytest.mark.asyncio
async def test_agentic_ask_budget_exhausted_progress_events(
    mock_hybrid_search: AsyncMock,
    mock_llm_gateway: MagicMock,
    mock_get_active_prompt: AsyncMock,
    mock_critique_generator: AsyncMock,
    mock_query_refiner: MagicMock,
    rag_validator: RagContentValidator,
    secret_pii_scanner: SecretPiiScanner,
):
    # 1회차 완료 후 2회차에서 max_iterations=2로 인해 generating 직후 budget 소진되는 경우
    mock_critique_generator.generate.return_value = Critique.of(
        answered=False,
        missing=["남은 정보"],
        next_query="남은 정보 검색",
        confidence=0.4,
    )

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
        on_progress=on_progress,
    )

    command = AgenticAskCommand(
        question="질문",
        budget=IterationBudget.of(max_iterations=2, token_budget=1000, timeout_ms=5000),
        confidence_threshold=0.8,
    )

    chunks = [c async for c in use_case.execute(command)]
    assert "".join(chunks) == "테스트 답변입니다."

    assert len(events) == 7
    assert events[0] == {"iteration": 1, "phase": "searching", "confidence": 0.0, "missing": []}
    assert events[1] == {"iteration": 1, "phase": "generating", "confidence": 0.0, "missing": []}
    assert events[2] == {"iteration": 1, "phase": "critiquing", "confidence": 0.0, "missing": []}
    assert events[3] == {
        "iteration": 1,
        "phase": "refining",
        "confidence": 0.4,
        "missing": ["남은 정보"],
    }
    assert events[4] == {
        "iteration": 2,
        "phase": "searching",
        "confidence": 0.4,
        "missing": ["남은 정보"],
    }
    assert events[5] == {
        "iteration": 2,
        "phase": "generating",
        "confidence": 0.4,
        "missing": ["남은 정보"],
    }
    assert events[6] == {
        "iteration": 2,
        "phase": "generating",
        "confidence": 0.4,
        "missing": ["남은 정보"],
    }


@pytest.mark.asyncio
async def test_agentic_ask_progress_callback_error_does_not_break_generation(
    mock_hybrid_search: AsyncMock,
    mock_llm_gateway: MagicMock,
    mock_get_active_prompt: AsyncMock,
    mock_critique_generator: AsyncMock,
    mock_query_refiner: MagicMock,
    rag_validator: RagContentValidator,
    secret_pii_scanner: SecretPiiScanner,
):
    mock_critique_generator.generate.return_value = Critique.of(
        answered=True, missing=[], next_query="", confidence=0.9
    )

    async def failing_on_progress(_data: dict) -> None:
        raise ConnectionError("Redis connection lost")

    use_case = AgenticAskUseCase(
        hybrid_search=mock_hybrid_search,
        llm_gateway=mock_llm_gateway,
        get_active_prompt=mock_get_active_prompt,
        critique_generator=mock_critique_generator,
        query_refiner=mock_query_refiner,
        rag_validator=rag_validator,
        secret_pii_scanner=secret_pii_scanner,
        on_progress=failing_on_progress,
    )

    command = AgenticAskCommand(
        question="질문",
        budget=IterationBudget.of(max_iterations=2, token_budget=1000, timeout_ms=5000),
        confidence_threshold=0.8,
    )

    chunks = [c async for c in use_case.execute(command)]
    assert "".join(chunks) == "테스트 답변입니다."
