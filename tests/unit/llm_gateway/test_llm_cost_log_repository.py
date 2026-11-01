from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from ai_service.llm_gateway.repository import LlmCostLogRepository
from ai_service.llm_gateway.schemas import ModelCostSum


@pytest.mark.asyncio
async def test_sum_by_model_aggregates_tokens_and_call_count() -> None:
    mock_cursor = MagicMock()
    mock_cursor.to_list = AsyncMock(
        return_value=[
            {
                "_id": "gpt-4o",
                "totalCostUsd": 1.25,
                "totalPromptTokens": 1000,
                "totalCompletionTokens": 500,
                "callCount": 10,
            },
            {
                "_id": "claude-3-5-sonnet",
                "totalCostUsd": 0.50,
                "totalPromptTokens": 400,
                "totalCompletionTokens": 200,
                "callCount": 2,
            },
        ]
    )

    mock_collection = MagicMock()
    mock_collection.aggregate.return_value = mock_cursor

    mock_db = MagicMock()
    mock_db.__getitem__.return_value = mock_collection

    repo = LlmCostLogRepository(mock_db)

    from_date = datetime(2026, 11, 1, 0, 0, tzinfo=UTC)
    to_date = datetime(2026, 11, 2, 0, 0, tzinfo=UTC)

    result = await repo.sum_by_model(from_date, to_date)

    mock_collection.aggregate.assert_called_once_with(
        [
            {"$match": {"createdAt": {"$gte": from_date, "$lte": to_date}}},
            {
                "$group": {
                    "_id": "$model",
                    "totalCostUsd": {"$sum": "$costUsd"},
                    "totalPromptTokens": {"$sum": "$promptTokens"},
                    "totalCompletionTokens": {"$sum": "$completionTokens"},
                    "callCount": {"$sum": 1},
                }
            },
            {"$sort": {"totalCostUsd": -1}},
        ]
    )

    assert result == [
        ModelCostSum(
            model="gpt-4o",
            total_cost_usd=1.25,
            prompt_tokens=1000,
            completion_tokens=500,
            call_count=10,
        ),
        ModelCostSum(
            model="claude-3-5-sonnet",
            total_cost_usd=0.50,
            prompt_tokens=400,
            completion_tokens=200,
            call_count=2,
        ),
    ]


@pytest.mark.asyncio
async def test_sum_by_model_handles_missing_fields_safely() -> None:
    mock_cursor = MagicMock()
    mock_cursor.to_list = AsyncMock(
        return_value=[
            {
                "_id": "gpt-4o",
            }
        ]
    )

    mock_collection = MagicMock()
    mock_collection.aggregate.return_value = mock_cursor

    mock_db = MagicMock()
    mock_db.__getitem__.return_value = mock_collection

    repo = LlmCostLogRepository(mock_db)

    from_date = datetime(2026, 11, 1, 0, 0, tzinfo=UTC)
    to_date = datetime(2026, 11, 2, 0, 0, tzinfo=UTC)

    result = await repo.sum_by_model(from_date, to_date)

    assert result == [
        ModelCostSum(
            model="gpt-4o",
            total_cost_usd=0.0,
            prompt_tokens=0,
            completion_tokens=0,
            call_count=0,
        )
    ]


@pytest.mark.asyncio
async def test_sum_by_model_returns_empty_list_when_no_logs() -> None:
    mock_cursor = MagicMock()
    mock_cursor.to_list = AsyncMock(return_value=[])

    mock_collection = MagicMock()
    mock_collection.aggregate.return_value = mock_cursor

    mock_db = MagicMock()
    mock_db.__getitem__.return_value = mock_collection

    repo = LlmCostLogRepository(mock_db)

    from_date = datetime(2026, 11, 1, 0, 0, tzinfo=UTC)
    to_date = datetime(2026, 11, 2, 0, 0, tzinfo=UTC)

    result = await repo.sum_by_model(from_date, to_date)
    assert result == []


@pytest.mark.asyncio
async def test_sum_by_model_handles_null_values_for_token_fields() -> None:
    mock_cursor = MagicMock()
    mock_cursor.to_list = AsyncMock(
        return_value=[
            {
                "_id": "claude-3-5-sonnet",
                "totalCostUsd": None,
                "totalPromptTokens": None,
                "totalCompletionTokens": None,
                "callCount": None,
            }
        ]
    )

    mock_collection = MagicMock()
    mock_collection.aggregate.return_value = mock_cursor

    mock_db = MagicMock()
    mock_db.__getitem__.return_value = mock_collection

    repo = LlmCostLogRepository(mock_db)

    from_date = datetime(2026, 11, 1, 0, 0, tzinfo=UTC)
    to_date = datetime(2026, 11, 2, 0, 0, tzinfo=UTC)

    result = await repo.sum_by_model(from_date, to_date)
    assert result == [
        ModelCostSum(
            model="claude-3-5-sonnet",
            total_cost_usd=0.0,
            prompt_tokens=0,
            completion_tokens=0,
            call_count=0,
        )
    ]

