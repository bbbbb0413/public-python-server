from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from ai_service.knowledge.application.ingest_document_use_case import IngestDocumentUseCase
from ai_service.knowledge.dependencies import get_document_repository, get_ingest_use_case
from ai_service.knowledge.repository import DocumentRepository
from ai_service.knowledge.schemas import Document, IngestDocumentCommand
from ai_service.main import app


@pytest.fixture(autouse=True)
def _clear_overrides():
    yield
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_retry_document_ingest_success() -> None:
    repo_mock = MagicMock(spec=DocumentRepository)
    existing_doc = Document(
        id="doc-failed-1",
        file_name="sample.txt",
        mime_type="text/plain",
        status="failed",
        chunk_count=0,
        created_at=datetime.now(UTC),
    )
    repo_mock.find_by_id = AsyncMock(return_value=existing_doc)
    repo_mock.get_original_file = AsyncMock(
        return_value=(b"sample content", "sample.txt", "text/plain")
    )

    processed_doc = Document(
        id="doc-failed-1",
        file_name="sample.txt",
        mime_type="text/plain",
        status="processed",
        chunk_count=3,
        created_at=existing_doc.created_at,
    )
    use_case_mock = MagicMock(spec=IngestDocumentUseCase)
    use_case_mock.execute = AsyncMock(return_value=processed_doc)

    app.dependency_overrides[get_document_repository] = lambda: repo_mock
    app.dependency_overrides[get_ingest_use_case] = lambda: use_case_mock

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/knowledge/documents/doc-failed-1/retry")

    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "doc-failed-1"
    assert data["fileName"] == "sample.txt"
    assert data["mimeType"] == "text/plain"
    assert data["status"] == "processed"
    assert data["chunkCount"] == 3

    repo_mock.find_by_id.assert_awaited_once_with("doc-failed-1")
    repo_mock.get_original_file.assert_awaited_once_with("doc-failed-1")
    use_case_mock.execute.assert_awaited_once_with(
        IngestDocumentCommand(
            document_id="doc-failed-1",
            file_name="sample.txt",
            mime_type="text/plain",
            content=b"sample content",
        )
    )


@pytest.mark.asyncio
async def test_retry_document_ingest_not_found_when_doc_missing() -> None:
    repo_mock = MagicMock(spec=DocumentRepository)
    repo_mock.find_by_id = AsyncMock(return_value=None)
    use_case_mock = MagicMock(spec=IngestDocumentUseCase)

    app.dependency_overrides[get_document_repository] = lambda: repo_mock
    app.dependency_overrides[get_ingest_use_case] = lambda: use_case_mock

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/knowledge/documents/doc-not-found/retry")

    assert response.status_code == 404
    repo_mock.find_by_id.assert_awaited_once_with("doc-not-found")
    use_case_mock.execute.assert_not_called()


@pytest.mark.asyncio
async def test_retry_document_ingest_not_found_when_original_file_missing() -> None:
    repo_mock = MagicMock(spec=DocumentRepository)
    existing_doc = Document(
        id="doc-no-file",
        file_name="sample.txt",
        mime_type="text/plain",
        status="failed",
        chunk_count=0,
        created_at=datetime.now(UTC),
    )
    repo_mock.find_by_id = AsyncMock(return_value=existing_doc)
    repo_mock.get_original_file = AsyncMock(return_value=None)
    use_case_mock = MagicMock(spec=IngestDocumentUseCase)

    app.dependency_overrides[get_document_repository] = lambda: repo_mock
    app.dependency_overrides[get_ingest_use_case] = lambda: use_case_mock

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/knowledge/documents/doc-no-file/retry")

    assert response.status_code == 404
    repo_mock.find_by_id.assert_awaited_once_with("doc-no-file")
    repo_mock.get_original_file.assert_awaited_once_with("doc-no-file")
    use_case_mock.execute.assert_not_called()
