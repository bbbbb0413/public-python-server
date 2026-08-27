import re
import struct
import time
import uuid
from dataclasses import dataclass
from typing import Any, cast

from redis.asyncio import Redis


@dataclass(frozen=True)
class SemanticCacheHit:
    answer: str
    score: float
    sources: list[dict[str, Any]] | None = None


INDEX_NAME = "sem_cache_idx"
KEY_PREFIX = "sem:cache:"
INDEX_ALREADY_EXISTS_MESSAGE = "Index already exists"
_TAG_ESCAPE_PATTERN = re.compile(r"[,.<>{}\[\]\"':;!@#$%^&*()\-+=~| ]")


def _escape_tag_value(value: str) -> str:
    return _TAG_ESCAPE_PATTERN.sub(lambda m: f"\\{m.group(0)}", value)


class RedisSemanticCacheAdapter:
    """RediSearch(FT.CREATE/FT.SEARCH) 모듈이 활성화된 Redis가 필요하다.

    docker-compose의 기본 `redis:alpine` 이미지에는 이 모듈이 없다 —
    이 어댑터를 실제로 쓰려면 `redis/redis-stack-server`(또는 RediSearch 모듈이
    포함된 이미지)로 교체해야 한다. NestJS(v1) 구현도 동일한 전제를 두고 있었다.
    """

    def __init__(self, redis_client: Redis, embedding_dimension: int) -> None:
        self._redis = redis_client
        self._embedding_dimension = embedding_dimension
        self._index_ready = False

    async def ensure_index(self) -> None:
        try:
            await cast(Any, self._redis).execute_command(
                "FT.CREATE",
                INDEX_NAME,
                "ON",
                "HASH",
                "PREFIX",
                "1",
                KEY_PREFIX,
                "SCHEMA",
                "embedding",
                "VECTOR",
                "HNSW",
                "6",
                "TYPE",
                "FLOAT32",
                "DIM",
                str(self._embedding_dimension),
                "DISTANCE_METRIC",
                "COSINE",
                "tenant",
                "TAG",
            )
        except Exception as e:
            if INDEX_ALREADY_EXISTS_MESSAGE not in str(e):
                raise
        self._index_ready = True

    async def find_similar(
        self, embedding: list[float], threshold: float, tenant: str
    ) -> SemanticCacheHit | None:
        if not self._index_ready:
            return None
        blob = struct.pack(f"<{len(embedding)}f", *embedding)
        reply = await cast(Any, self._redis).execute_command(
            "FT.SEARCH",
            INDEX_NAME,
            f"(@tenant:{{{_escape_tag_value(tenant)}}})=>[KNN 1 @embedding $vec AS dist]",
            "PARAMS",
            "2",
            "vec",
            blob,
            "SORTBY",
            "dist",
            "RETURN",
            "2",
            "answer",
            "dist",
            "DIALECT",
            "2",
        )

        parsed = self._parse_knn_reply(reply)
        if parsed is None:
            return None

        score = 1 - parsed[1]
        return SemanticCacheHit(answer=parsed[0], score=score) if score >= threshold else None

    async def store(
        self,
        embedding: list[float],
        question: str,
        answer: str,
        ttl_seconds: int,
        tenant: str,
    ) -> None:
        if not self._index_ready:
            return
        key = f"{KEY_PREFIX}{uuid.uuid4()}"
        blob = struct.pack(f"<{len(embedding)}f", *embedding)
        await self._redis.hset(
            key,
            mapping={
                "embedding": blob,
                "answer": answer,
                "question": question,
                "tenant": tenant,
                "createdAt": str(int(time.time() * 1000)),
            },
        )
        await self._redis.expire(key, ttl_seconds)

    @staticmethod
    def _parse_knn_reply(reply: object) -> tuple[str, float] | None:
        if not isinstance(reply, list) or not reply or not reply[0]:
            return None

        fields = reply[2] if len(reply) > 2 else None
        if not isinstance(fields, list):
            return None

        result: dict[str, str] = {}
        for i in range(0, len(fields), 2):
            key = fields[i]
            value = fields[i + 1]
            result[key.decode() if isinstance(key, bytes) else key] = (
                value.decode() if isinstance(value, bytes) else value
            )

        if "answer" not in result or "dist" not in result:
            return None

        return result["answer"], float(result["dist"])
