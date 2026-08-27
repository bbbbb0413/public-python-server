import asyncio
import base64
import json
import logging
from dataclasses import dataclass
from typing import Any

from aiokafka import AIOKafkaConsumer
from redis.asyncio import Redis

from ai_service.core.events import JobEventPublisher
from ai_service.knowledge.knowledge_composition import KnowledgeComposition
from ai_service.knowledge.schemas import IngestDocumentCommand

logger = logging.getLogger(__name__)

INGEST_REQUESTED_TOPIC = "ai.knowledge.ingest.requested"
CONSUMER_GROUP_ID = "ai-service-knowledge"


@dataclass(frozen=True)
class IngestRequestedMessage:
    job_id: str
    file_name: str
    mime_type: str

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "IngestRequestedMessage":
        return cls(
            job_id=payload["jobId"],
            file_name=payload["fileName"],
            mime_type=payload["mimeType"],
        )


class IngestRequestedConsumer:
    def __init__(
        self, brokers: str, redis_client: Redis, composition: KnowledgeComposition
    ) -> None:
        self._brokers = brokers
        self._redis = redis_client
        self._publisher = JobEventPublisher(redis_client)
        self._composition = composition
        self._consumer: AIOKafkaConsumer | None = None
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        self._consumer = AIOKafkaConsumer(
            INGEST_REQUESTED_TOPIC,
            bootstrap_servers=self._brokers,
            group_id=CONSUMER_GROUP_ID,
            enable_auto_commit=True,
        )
        await self._consumer.start()
        self._task = asyncio.create_task(self._consume_loop())
        logger.info("IngestRequestedConsumer started: topic=%s", INGEST_REQUESTED_TOPIC)

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
        if self._consumer is not None:
            await self._consumer.stop()
        logger.info("IngestRequestedConsumer stopped")

    async def _consume_loop(self) -> None:
        assert self._consumer is not None
        async for record in self._consumer:
            await self._handle_record(record.value)

    async def _handle_record(self, raw_value: bytes) -> None:
        payload = json.loads(raw_value.decode("utf-8"))
        message = IngestRequestedMessage.from_payload(payload)

        try:
            await self._process(message)
        except Exception as exc:
            logger.exception("문서 인제스트 잡 처리 실패: jobId=%s", message.job_id)
            await self._publisher.publish_error(message.job_id, str(exc))
        finally:
            await self._redis.delete(self._file_key(message.job_id))

    async def _process(self, message: IngestRequestedMessage) -> None:
        encoded_content = await self._redis.get(self._file_key(message.job_id))
        if encoded_content is None:
            raise ValueError(f"업로드된 파일을 찾을 수 없습니다: jobId={message.job_id}")

        content = base64.b64decode(encoded_content)
        document = await self._composition.ingest_use_case.execute(
            IngestDocumentCommand(
                file_name=message.file_name,
                mime_type=message.mime_type,
                content=content,
            )
        )

        await self._publisher.publish_done(
            message.job_id,
            {"documentId": document.id, "chunkCount": document.chunk_count},
        )

    @staticmethod
    def _file_key(job_id: str) -> str:
        return f"ai:job:{job_id}:file"
