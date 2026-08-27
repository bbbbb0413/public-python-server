from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from motor.motor_asyncio import AsyncIOMotorClient
from redis.asyncio import Redis

from ai_service.core.config import get_settings
from ai_service.knowledge.infrastructure.messaging.ingest_requested_consumer import (
    IngestRequestedConsumer,
)
from ai_service.knowledge.knowledge_composition import build_knowledge_composition
from ai_service.knowledge.router import router as knowledge_router
from ai_service.llm_gateway.circuit_breaker import CIRCUIT_BREAKER_DB
from ai_service.llm_gateway.router import router as llm_gateway_router
from ai_service.observability.router import router as observability_router
from ai_service.prompt.router import router as prompt_router
from ai_service.rag.infrastructure.messaging.ask_requested_consumer import (
    AskRequestedConsumer,
)
from ai_service.rag.rag_composition import build_rag_composition
from ai_service.rag.router import router as rag_router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()

    mongo_client: AsyncIOMotorClient[dict[str, Any]] = AsyncIOMotorClient(
        settings.mongodb_vector_uri
    )
    redis_client: Redis = Redis(
        host=settings.redis_db_host, port=settings.redis_db_port, decode_responses=True
    )
    redis_cb_client: Redis = Redis(
        host=settings.redis_db_host,
        port=settings.redis_db_port,
        db=CIRCUIT_BREAKER_DB,
        decode_responses=True,
    )

    app.state.mongo_client = mongo_client
    app.state.mongo_db = mongo_client[settings.mongodb_db_name]
    app.state.redis_client = redis_client
    app.state.redis_cb_client = redis_cb_client

    rag_composition = await build_rag_composition(
        settings, app.state.mongo_db, redis_client, redis_cb_client
    )
    ask_requested_consumer = AskRequestedConsumer(
        settings.kafka_brokers, redis_client, rag_composition
    )
    await ask_requested_consumer.start()
    app.state.ask_requested_consumer = ask_requested_consumer

    knowledge_composition = await build_knowledge_composition(settings, app.state.mongo_db)
    app.state.vector_store = knowledge_composition.vector_store
    ingest_requested_consumer = IngestRequestedConsumer(
        settings.kafka_brokers, redis_client, knowledge_composition
    )
    await ingest_requested_consumer.start()
    app.state.ingest_requested_consumer = ingest_requested_consumer

    try:
        yield
    finally:
        await ask_requested_consumer.stop()
        await ingest_requested_consumer.stop()
        mongo_client.close()
        await redis_client.aclose()
        await redis_cb_client.aclose()


app = FastAPI(title="AI Service", version="0.1.0", lifespan=lifespan)

app.include_router(llm_gateway_router)
app.include_router(prompt_router)
app.include_router(observability_router)
app.include_router(rag_router)
app.include_router(knowledge_router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


__all__ = ["app"]
