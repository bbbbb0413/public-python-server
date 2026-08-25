import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from redis.asyncio import Redis

from ai_service.shared_kernel.messaging.job_event_publisher import JobEventPublisher


@pytest.mark.asyncio
async def test_publish_progress_adds_to_stream():
    redis_mock = MagicMock(spec=Redis)
    redis_mock.xadd = AsyncMock()
    publisher = JobEventPublisher(redis_mock)

    progress_data = {
        "iteration": 1,
        "phase": "searching",
        "confidence": 0.0,
        "missing": [],
    }
    await publisher.publish_progress("job-123", progress_data)

    redis_mock.xadd.assert_awaited_once_with(
        "ai:job:job-123:events",
        {"type": "progress", "data": json.dumps(progress_data)},
    )


@pytest.mark.asyncio
async def test_publish_done_without_data_adds_to_stream():
    redis_mock = MagicMock(spec=Redis)
    redis_mock.xadd = AsyncMock()
    publisher = JobEventPublisher(redis_mock)

    await publisher.publish_done("job-123")

    redis_mock.xadd.assert_awaited_once_with(
        "ai:job:job-123:events",
        {"type": "done"},
    )


@pytest.mark.asyncio
async def test_publish_done_with_data_adds_to_stream():
    redis_mock = MagicMock(spec=Redis)
    redis_mock.xadd = AsyncMock()
    publisher = JobEventPublisher(redis_mock)

    done_data = {"confidence": 0.85, "missing": ["추가 설명"]}
    await publisher.publish_done("job-123", done_data)

    redis_mock.xadd.assert_awaited_once_with(
        "ai:job:job-123:events",
        {"type": "done", "data": json.dumps(done_data)},
    )

