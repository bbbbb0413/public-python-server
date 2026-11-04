from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from ai_service.observability.repository import RagasEvaluationRepository
from ai_service.observability.schemas import RagasEvaluation


def _make_repo() -> tuple[RagasEvaluationRepository, MagicMock]:
    repo = RagasEvaluationRepository.__new__(RagasEvaluationRepository)
    mock_collection = MagicMock()
    repo._collection = mock_collection
    return repo, mock_collection


@pytest.mark.asyncio
async def test_persist_stores_answer_and_contexts() -> None:
    repo, collection = _make_repo()
    collection.insert_one = AsyncMock()

    eval_item = RagasEvaluation(
        trace_id="trace-123",
        question="질문입니다",
        faithfulness=0.85,
        answer_relevancy=0.90,
        context_precision=0.75,
        sampled_at=datetime(2026, 1, 1, tzinfo=UTC),
        answer="답변 내용입니다.",
        contexts=["근거 컨텍스트 1", "근거 컨텍스트 2"],
    )

    await repo.persist(eval_item)

    collection.insert_one.assert_awaited_once_with(
        {
            "traceId": "trace-123",
            "question": "질문입니다",
            "faithfulness": 0.85,
            "answerRelevancy": 0.90,
            "contextPrecision": 0.75,
            "sampledAt": datetime(2026, 1, 1, tzinfo=UTC),
            "answer": "답변 내용입니다.",
            "contexts": ["근거 컨텍스트 1", "근거 컨텍스트 2"],
        }
    )


@pytest.mark.asyncio
async def test_find_recent_restores_answer_and_contexts() -> None:
    repo, collection = _make_repo()
    mock_cursor = MagicMock()
    collection.find.return_value = mock_cursor
    mock_cursor.sort.return_value = mock_cursor
    mock_cursor.limit.return_value = mock_cursor
    mock_cursor.to_list = AsyncMock(
        return_value=[
            {
                "traceId": "t1",
                "question": "q1",
                "faithfulness": 0.9,
                "answerRelevancy": 0.8,
                "contextPrecision": 0.7,
                "sampledAt": datetime(2026, 1, 2, tzinfo=UTC),
                "answer": "생성된 답변",
                "contexts": ["컨텍스트 A"],
            }
        ]
    )

    results = await repo.find_recent(10)

    assert len(results) == 1
    assert results[0].answer == "생성된 답변"
    assert results[0].contexts == ["컨텍스트 A"]


@pytest.mark.asyncio
async def test_find_recent_fallbacks_for_legacy_documents_without_answer_or_contexts() -> None:
    repo, collection = _make_repo()
    mock_cursor = MagicMock()
    collection.find.return_value = mock_cursor
    mock_cursor.sort.return_value = mock_cursor
    mock_cursor.limit.return_value = mock_cursor
    mock_cursor.to_list = AsyncMock(
        return_value=[
            {
                "traceId": "legacy-t1",
                "question": "과거 질문",
                "faithfulness": 0.5,
                "answerRelevancy": 0.5,
                "contextPrecision": 0.5,
                "sampledAt": datetime(2026, 1, 1, tzinfo=UTC),
                # answer, contexts 키 부재
            }
        ]
    )

    results = await repo.find_recent(10)

    assert len(results) == 1
    assert results[0].trace_id == "legacy-t1"
    assert results[0].answer == ""
    assert results[0].contexts == []
