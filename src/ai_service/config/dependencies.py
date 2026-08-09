from typing import Annotated, Any

from fastapi import Depends, Request
from motor.motor_asyncio import AsyncIOMotorDatabase
from redis.asyncio import Redis


def get_mongo_db(request: Request) -> AsyncIOMotorDatabase[dict[str, Any]]:
    return request.app.state.mongo_db  # type: ignore[no-any-return]


def get_redis_client(request: Request) -> Redis:
    return request.app.state.redis_client  # type: ignore[no-any-return]


def get_circuit_breaker_redis_client(request: Request) -> Redis:
    return request.app.state.redis_cb_client  # type: ignore[no-any-return]


MongoDbDep = Annotated[AsyncIOMotorDatabase[dict[str, Any]], Depends(get_mongo_db)]
RedisClientDep = Annotated[Redis, Depends(get_redis_client)]
CircuitBreakerRedisClientDep = Annotated[Redis, Depends(get_circuit_breaker_redis_client)]
