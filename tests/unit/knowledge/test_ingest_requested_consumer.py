import base64
import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from redis.asyncio import Redis

from ai_service.knowledge.application.ingest_document_use_case import IngestDocumentUseCase
from ai_service.knowledge.infrastructure.messaging.ingest_requested_consumer import (
    IngestRequestedConsumer,
    IngestRequestedMessage,
)
from ai_service.knowledge.knowledge_composition import KnowledgeComposition
from ai_service.knowledge.schemas import Document


@pytest.mark.asyncio
async def test_ingest_requested_consumer_publishes_progress_and_done():
    redis_mock = MagicMock(spec=Redis)
    redis_mock.xadd = AsyncMock()
    encoded_file = base64.b64encode(b"Sample file content").decode("utf-8")
    redis_mock.get = AsyncMock(return_value=encoded_file)
    redis_mock.delete = AsyncMock()

    use_case_mock = MagicMock(spec=IngestDocumentUseCase)

    async def fake_execute(command, on_progress=None):
        if on_progress:
            await on_progress("extract", 20)
            await on_progress("chunk", 45)
            await on_progress("embed", 70)
            await on_progress("index", 90)
        return Document(
            id="doc-999",
            file_name=command.file_name,
            mime_type=command.mime_type,
            chunk_count=3,
            status="processed",
            created_at=datetime.now(UTC),
        )

    use_case_mock.execute.side_effect = fake_execute

    composition = KnowledgeComposition(
        ingest_use_case=use_case_mock,
        document_repo=MagicMock(),
        vector_store=MagicMock(),
    )

    consumer = IngestRequestedConsumer("localhost:9092", redis_mock, composition)

    message = IngestRequestedMessage(
        job_id="test-job-ingest-1",
        file_name="test.txt",
        mime_type="text/plain",
    )

    await consumer._process(message)

    # progress 이벤트 4개가 순차적으로 발행되었는지 확인
    progress_calls = [
        call
        for call in redis_mock.xadd.await_args_list
        if call.args[1].get("type") == "progress"
    ]
    assert len(progress_calls) == 4

    assert json.loads(progress_calls[0].args[1]["data"]) == {"step": "extract", "progress": 20}
    assert json.loads(progress_calls[1].args[1]["data"]) == {"step": "chunk", "progress": 45}
    assert json.loads(progress_calls[2].args[1]["data"]) == {"step": "embed", "progress": 70}
    assert json.loads(progress_calls[3].args[1]["data"]) == {"step": "index", "progress": 90}

    # done 이벤트 발행 확인
    done_calls = [
        call
        for call in redis_mock.xadd.await_args_list
        if call.args[1].get("type") == "done"
    ]
    assert len(done_calls) == 1
    assert json.loads(done_calls[0].args[1]["data"]) == {
        "documentId": "doc-999",
        "chunkCount": 3,
    }


@pytest.mark.asyncio
async def test_ingest_requested_consumer_publishes_error_on_missing_file():
    redis_mock = MagicMock(spec=Redis)
    redis_mock.xadd = AsyncMock()
    redis_mock.get = AsyncMock(return_value=None)
    redis_mock.delete = AsyncMock()

    composition = KnowledgeComposition(
        ingest_use_case=MagicMock(spec=IngestDocumentUseCase),
        document_repo=MagicMock(),
        vector_store=MagicMock(),
    )

    consumer = IngestRequestedConsumer("localhost:9092", redis_mock, composition)

    raw_payload = json.dumps(
        {"jobId": "test-job-fail", "fileName": "missing.txt", "mimeType": "text/plain"}
    ).encode("utf-8")

    await consumer._handle_record(raw_payload)

    # error 이벤트 발행 확인
    error_calls = [
        call
        for call in redis_mock.xadd.await_args_list
        if call.args[1].get("type") == "error"
    ]
    assert len(error_calls) == 1
    assert "업로드된 파일을 찾을 수 없습니다" in error_calls[0].args[1]["data"]
