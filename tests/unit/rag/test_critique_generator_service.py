from collections.abc import AsyncIterator
from unittest.mock import MagicMock

import pytest

from ai_service.core.config import Settings
from ai_service.knowledge.schemas import (
    SimilaritySearchResult,
    VectorDocumentMetadata,
)
from ai_service.llm_gateway.application.llm_gateway_service import LlmGatewayService
from ai_service.rag.application.critique_generator_service import CritiqueGeneratorService


def _create_fake_chunks() -> list[SimilaritySearchResult]:
    return [
        SimilaritySearchResult(
            text="문서 본문 내용입니다.",
            score=0.9,
            metadata=VectorDocumentMetadata(
                document_id="doc-1",
                file_name="guide.pdf",
                chunk_index=0,
            ),
        )
    ]


def _make_service(mock_stream_responses: list[str]) -> CritiqueGeneratorService:
    mock_gateway = MagicMock(spec=LlmGatewayService)

    async def fake_stream(_command) -> AsyncIterator[str]:
        for token in mock_stream_responses:
            yield token

    mock_gateway.stream.side_effect = fake_stream
    settings = Settings(openai_api_key="")
    return CritiqueGeneratorService(llm_gateway=mock_gateway, settings=settings)


@pytest.mark.asyncio
async def test_critique_generator_normal_json():
    # 정상 JSON 응답 파싱
    raw_response = (
        '{"answered": true, "missing": [], "nextQuery": "", "confidence": 0.85}'
    )
    service = _make_service([raw_response])
    critique = await service.generate("질문", "답변", _create_fake_chunks())

    assert critique.is_satisfied(0.8) is True
    assert critique.get_confidence() == 0.85
    assert critique.get_missing() == []
    assert critique.get_next_query() == ""


@pytest.mark.asyncio
async def test_critique_generator_markdown_wrapped_json():
    # 마크다운 코드블록 내 JSON 응답 파싱
    raw_response = (
        "```json\n"
        '{"answered": false, "missing": ["상세 규정"], '
        '"nextQuery": "상세 규정 안내", "confidence": 0.4}\n'
        "```"
    )
    service = _make_service([raw_response])
    critique = await service.generate("질문", "답변", _create_fake_chunks())

    assert critique.is_satisfied(0.6) is False
    assert critique.get_confidence() == 0.4
    assert critique.get_missing() == ["상세 규정"]
    assert critique.get_next_query() == "상세 규정 안내"


@pytest.mark.asyncio
async def test_critique_generator_fallback_on_invalid_json():
    # 유효하지 않은 JSON 응답 시 안전한 폴백 검증
    raw_response = "죄송하지만 이 질문에 대해서는 평가할 수 없습니다. 형식이 잘못되었습니다."
    service = _make_service([raw_response])
    critique = await service.generate("질문", "답변", _create_fake_chunks())

    assert critique.is_satisfied(0.1) is False
    assert critique.get_confidence() == 0.0
    assert critique.get_missing() == ["비평 파싱 실패"]
    assert critique.get_next_query() == ""


@pytest.mark.asyncio
async def test_critique_generator_fallback_on_empty_string():
    # 빈 문자열 응답 시 안전한 폴백 검증
    service = _make_service([""])
    critique = await service.generate("질문", "답변", _create_fake_chunks())

    assert critique.is_satisfied(0.1) is False
    assert critique.get_confidence() == 0.0
    assert critique.get_missing() == ["비평 파싱 실패"]
    assert critique.get_next_query() == ""


@pytest.mark.asyncio
async def test_critique_generator_fallback_on_broken_json():
    # 깨진 JSON 응답 시 안전한 폴백 검증
    raw_response = '{"answered": true, "missing": ["누락"], "confidence": '
    service = _make_service([raw_response])
    critique = await service.generate("질문", "답변", _create_fake_chunks())

    assert critique.is_satisfied(0.1) is False
    assert critique.get_confidence() == 0.0
    assert critique.get_missing() == ["비평 파싱 실패"]
    assert critique.get_next_query() == ""


@pytest.mark.asyncio
async def test_critique_generator_missing_confidence_field():
    # confidence 필드 누락 시 0.0으로 기본 처리
    raw_response = '{"answered": true, "missing": [], "nextQuery": ""}'
    service = _make_service([raw_response])
    critique = await service.generate("질문", "답변", _create_fake_chunks())

    assert critique.get_confidence() == 0.0
    assert critique.get_missing() == []
    assert critique.get_next_query() == ""


@pytest.mark.asyncio
async def test_critique_generator_invalid_confidence_values():
    # confidence가 음수 또는 1 초과 또는 잘못된 타입인 경우 0.0으로 정규화
    test_cases = [
        '{"answered": true, "confidence": -0.5}',
        '{"answered": true, "confidence": 1.5}',
        '{"answered": true, "confidence": "high"}',
        '{"answered": true, "confidence": None}',
    ]

    for raw in test_cases:
        service = _make_service([raw])
        critique = await service.generate("질문", "답변", _create_fake_chunks())
        assert critique.get_confidence() == 0.0
