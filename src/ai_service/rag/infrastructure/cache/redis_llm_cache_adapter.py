from redis.asyncio import Redis


class RedisLlmCacheAdapter:
    def __init__(self, redis_client: Redis) -> None:
        self._redis = redis_client

    async def get(self, key: str) -> str | None:
        value = await self._redis.get(key)
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return value

    async def set_with_ttl(self, key: str, value: str, ttl_seconds: int) -> None:
        await self._redis.set(key, value, ex=ttl_seconds)

    async def invalidate(self, key: str) -> None:
        await self._redis.delete(key)
