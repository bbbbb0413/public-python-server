import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any

from aiokafka import AIOKafkaConsumer
from redis.asyncio import Redis

from ai_service.core.events import JobEventPublisher
from ai_service.rag.application.ask_command import AskCommand
from ai_service.rag.application.command.agentic_ask_command import AgenticAskCommand
from ai_service.rag.rag_composition import RagComposition
from ai_service.rag.schemas import ConversationSession

logger = logging.getLogger(__name__)

ASK_REQUESTED_TOPIC = "ai.rag.ask.requested"
CONSUMER_GROUP_ID = "ai-service-rag"

_SOURCES_PREFIX = "__SOURCES:"
_DEFAULT_ASK_TOP_K = 15
_DEFAULT_AGENTIC_TOP_K = 5


@dataclass(frozen=True)
class AskRequestedMessage:
    job_id: str
    user_id: str
    question: str
    top_k: int | None = None
    use_hyde: bool | None = None
    session_id: str | None = None
    conversation_history: list[dict[str, str]] | None = None

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "AskRequestedMessage":
        return cls(
            job_id=payload["jobId"],
            user_id=payload["userId"],
            question=payload["question"],
            top_k=payload.get("topK"),
            use_hyde=payload.get("useHyde"),
            session_id=payload.get("sessionId"),
            conversation_history=payload.get("conversationHistory"),
        )


class AskRequestedConsumer:
    def __init__(self, brokers: str, redis_client: Redis, composition: RagComposition) -> None:
        self._brokers = brokers
        self._redis = redis_client
        self._publisher = JobEventPublisher(redis_client)
        self._composition = composition
        self._consumer: AIOKafkaConsumer | None = None
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        self._consumer = AIOKafkaConsumer(
            ASK_REQUESTED_TOPIC,
            bootstrap_servers=self._brokers,
            group_id=CONSUMER_GROUP_ID,
            enable_auto_commit=True,
        )
        await self._consumer.start()
        self._task = asyncio.create_task(self._consume_loop())
        logger.info("AskRequestedConsumer started: topic=%s", ASK_REQUESTED_TOPIC)

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
        if self._consumer is not None:
            await self._consumer.stop()
        logger.info("AskRequestedConsumer stopped")

    async def _consume_loop(self) -> None:
        assert self._consumer is not None
        async for record in self._consumer:
            await self._handle_record(record.value)

    async def _handle_record(self, raw_value: bytes) -> None:
        payload = json.loads(raw_value.decode("utf-8"))
        message = AskRequestedMessage.from_payload(payload)

        try:
            await self._process(message)
        except Exception as exc:
            logger.exception("ask job 처리 실패: jobId=%s", message.job_id)
            await self._publisher.publish_error(message.job_id, str(exc))

    async def _process(self, message: AskRequestedMessage) -> None:
        composition = self._composition

        if composition.guardrail_enabled:
            verdict = composition.rag_validator.inspect_input(message.question)
            if not verdict.is_allowed():
                logger.warning("프롬프트 인젝션 차단: %s", verdict.get_reason())
                await self._publisher.publish_error(
                    message.job_id, "요청이 보안 정책에 의해 차단되었습니다."
                )
                return

        session = await self._resolve_session(message)
        if session is not None:
            await self._publisher.publish_session(message.job_id, session.get_session_id())

        complexity = composition.query_complexity_router.route(message.question)
        use_hyde = (
            message.use_hyde
            if message.use_hyde is not None
            else self._should_use_hyde(message.question, composition.hyde_max_query_words)
        )

        last_confidence: float | None = None
        last_missing: list[str] | None = None

        if complexity == "complex":
            async def on_progress(data: dict[str, Any]) -> None:
                nonlocal last_confidence, last_missing
                if "confidence" in data:
                    last_confidence = data["confidence"]
                if "missing" in data:
                    last_missing = data["missing"]
                await self._publisher.publish_progress(message.job_id, data)

            stream = composition.agentic_ask_use_case.execute(
                AgenticAskCommand(
                    question=message.question,
                    budget=composition.budget,
                    top_k=message.top_k or _DEFAULT_AGENTIC_TOP_K,
                    confidence_threshold=composition.confidence_threshold,
                    user_id=message.user_id,
                    conversation_history=message.conversation_history,
                    use_hyde=use_hyde,
                    on_progress=on_progress,
                )
            )
        else:
            stream = composition.ask_use_case.execute(
                AskCommand(
                    question=message.question,
                    top_k=message.top_k or _DEFAULT_ASK_TOP_K,
                    use_hyde=use_hyde,
                    user_id=message.user_id,
                    conversation_history=message.conversation_history,
                    session_id=session.get_session_id() if session else None,
                )
            )

        collected: list[str] = []
        sources: list[dict[str, Any]] | None = None
        async for chunk in stream:
            is_cancelled = await self._redis.get(f"job:{message.job_id}:cancelled")
            if is_cancelled:
                logger.info("ask job 취소 감지됨: jobId=%s", message.job_id)
                break

            if chunk.startswith(_SOURCES_PREFIX):
                sources = json.loads(chunk[len(_SOURCES_PREFIX) :])
                await self._publisher.publish_sources(message.job_id, sources)
            else:
                safe = composition.secret_pii_scanner.mask(chunk)
                collected.append(safe)
                await self._publisher.publish_token(message.job_id, safe)

        if session is not None and collected:
            full_response = "".join(collected)
            updated = session.append_turn(message.question, full_response, sources=sources)
            await composition.session_repo.update(updated)

        done_data: dict[str, Any] | None = None
        if last_confidence is not None and last_missing is not None:
            done_data = {
                "confidence": last_confidence,
                "missing": last_missing,
            }

        await self._publisher.publish_done(message.job_id, done_data)

    async def _resolve_session(self, message: AskRequestedMessage) -> ConversationSession | None:
        session_repo = self._composition.session_repo

        if message.session_id:
            return await session_repo.find_by_id(message.session_id)
        if message.user_id:
            session = ConversationSession.create(message.user_id, message.question)
            return await session_repo.persist(session)
        return None

    @staticmethod
    def _should_use_hyde(question: str, max_words: int) -> bool:
        return len([w for w in question.strip().split() if w]) <= max_words
