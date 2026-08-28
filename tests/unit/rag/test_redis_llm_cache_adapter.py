from unittest.mock import AsyncMock

from ai_service.rag.infrastructure.cache.redis_llm_cache_adapter import RedisLlmCacheAdapter


class TestRedisLlmCacheAdapter:
    async def test_redis가_bytes를_반환해도_str로_디코드해서_돌려준다(self) -> None:
        mock_redis = AsyncMock()
        mock_redis.get.return_value = b"cached answer"
        adapter = RedisLlmCacheAdapter(mock_redis)

        result = await adapter.get("some-key")

        assert result == "cached answer"
        assert isinstance(result, str)

    async def test_redis_client가_decode_responses로_이미_str을_반환하면_그대로_돌려준다(
        self,
    ) -> None:
        mock_redis = AsyncMock()
        mock_redis.get.return_value = "cached answer"
        adapter = RedisLlmCacheAdapter(mock_redis)

        result = await adapter.get("some-key")

        assert result == "cached answer"

    async def test_캐시_미스이면_None을_그대로_돌려준다(self) -> None:
        mock_redis = AsyncMock()
        mock_redis.get.return_value = None
        adapter = RedisLlmCacheAdapter(mock_redis)

        result = await adapter.get("some-key")

        assert result is None
