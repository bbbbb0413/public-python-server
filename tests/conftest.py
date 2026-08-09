import uuid
from collections.abc import AsyncIterator

import pytest
from motor.motor_asyncio import AsyncIOMotorClient
from redis.asyncio import Redis

TEST_MONGODB_URI = "mongodb://localhost:27017/?directConnection=true"
TEST_MONGODB_DB_PREFIX = "ai_service_test"
TEST_REDIS_DB = 13


@pytest.fixture
async def mongo_test_db() -> AsyncIterator[AsyncIOMotorClient]:
    client: AsyncIOMotorClient = AsyncIOMotorClient(TEST_MONGODB_URI, serverSelectionTimeoutMS=5000)
    db_name = f"{TEST_MONGODB_DB_PREFIX}_{uuid.uuid4().hex[:8]}"
    db = client[db_name]
    try:
        yield db
    finally:
        await client.drop_database(db_name)
        client.close()


@pytest.fixture
async def redis_test_client() -> AsyncIterator[Redis]:
    client: Redis = Redis(host="localhost", port=6379, db=TEST_REDIS_DB, decode_responses=True)
    try:
        await client.flushdb()
        yield client
    finally:
        await client.flushdb()
        await client.aclose()
