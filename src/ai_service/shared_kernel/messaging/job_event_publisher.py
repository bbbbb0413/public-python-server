import json
from typing import Any

from redis.asyncio import Redis


class JobEventPublisher:
    """ai:job:{jobId}:events 스트림에 XADD.

    Gateway의 RedisStreamsRelayService가 읽는 스키마와 동일하게 맞춘다.
    rag/knowledge 등 여러 잡 컨슈머가 공유한다.
    """

    def __init__(self, redis_client: Redis) -> None:
        self._redis = redis_client

    async def publish_session(self, job_id: str, session_id: str) -> None:
        await self._redis.xadd(self._stream_key(job_id), {"type": "session", "data": session_id})

    async def publish_token(self, job_id: str, text: str) -> None:
        await self._redis.xadd(self._stream_key(job_id), {"type": "token", "data": text})

    async def publish_sources(self, job_id: str, sources: list[dict[str, Any]]) -> None:
        await self._redis.xadd(
            self._stream_key(job_id), {"type": "sources", "data": json.dumps(sources)}
        )

    async def publish_progress(self, job_id: str, data: dict[str, Any]) -> None:
        await self._redis.xadd(
            self._stream_key(job_id), {"type": "progress", "data": json.dumps(data)}
        )

    async def publish_done(self, job_id: str, data: dict[str, Any] | None = None) -> None:
        if data is not None:
            await self._redis.xadd(
                self._stream_key(job_id), {"type": "done", "data": json.dumps(data)}
            )
        else:
            await self._redis.xadd(self._stream_key(job_id), {"type": "done"})

    async def publish_error(self, job_id: str, message: str) -> None:
        await self._redis.xadd(self._stream_key(job_id), {"type": "error", "data": message})

    @staticmethod
    def _stream_key(job_id: str) -> str:
        return f"ai:job:{job_id}:events"
