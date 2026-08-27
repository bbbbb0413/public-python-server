import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from redis.asyncio import Redis

from ai_service.core.events import JobEventPublisher


@pytest.mark.asyncio
async def test_publish_progress_adds_to_stream() -> None:
    redis_mock = MagicMock(spec=Redis)
    redis_mock.xadd = AsyncMock()
    publisher = JobEventPublisher(redis_mock)

    await publisher.publish_progress("job-123", {"progress": 50, "message": "진행 중"})

    redis_mock.xadd.assert_called_once_with(
        "ai:job:job-123:events",
        {"type": "progress", "data": json.dumps({"progress": 50, "message": "진행 중"})},
    )


@pytest.mark.asyncio
async def test_publish_done_adds_to_stream() -> None:
    redis_mock = MagicMock(spec=Redis)
    redis_mock.xadd = AsyncMock()
    publisher = JobEventPublisher(redis_mock)

    await publisher.publish_done("job-456", {"answer": "결과"})

    redis_mock.xadd.assert_called_once_with(
        "ai:job:job-456:events",
        {"type": "done", "data": json.dumps({"answer": "결과"})},
    )


@pytest.mark.asyncio
async def test_publish_error_adds_to_stream() -> None:
    redis_mock = MagicMock(spec=Redis)
    redis_mock.xadd = AsyncMock()
    publisher = JobEventPublisher(redis_mock)

    await publisher.publish_error("job-789", "오류 발생")

    redis_mock.xadd.assert_called_once_with(
        "ai:job:job-789:events",
        {"type": "error", "data": "오류 발생"},
    )
