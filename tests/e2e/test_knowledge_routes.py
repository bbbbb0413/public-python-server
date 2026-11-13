from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from ai_service.core.config import Settings, get_settings
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
async def test_retry_document_ingest_not_found_with_real_repo_when_id_invalid() -> None:
    repo = DocumentRepository.__new__(DocumentRepository)
    repo._collection = MagicMock()
    repo._collection.find_one = AsyncMock(return_value=None)
    mock_gridfs = MagicMock()
    repo._gridfs = mock_gridfs

    use_case_mock = MagicMock(spec=IngestDocumentUseCase)

    app.dependency_overrides[get_document_repository] = lambda: repo
    app.dependency_overrides[get_ingest_use_case] = lambda: use_case_mock

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/knowledge/documents/doc-not-found/retry")

    assert response.status_code == 404
    assert response.json()["detail"] == "문서를 찾을 수 없습니다: doc-not-found"
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


@pytest.mark.asyncio
async def test_retry_document_ingest_server_error_when_use_case_fails() -> None:
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

    use_case_mock = MagicMock(spec=IngestDocumentUseCase)
    use_case_mock.execute = AsyncMock(side_effect=RuntimeError("Embedding model timeout"))

    app.dependency_overrides[get_document_repository] = lambda: repo_mock
    app.dependency_overrides[get_ingest_use_case] = lambda: use_case_mock

    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/knowledge/documents/doc-failed-1/retry")

    assert response.status_code == 500


@pytest.mark.asyncio
async def test_retry_document_ingest_auth_with_admin_key() -> None:
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

    use_case_mock = MagicMock(spec=IngestDocumentUseCase)
    use_case_mock.execute = AsyncMock(return_value=existing_doc)

    app.dependency_overrides[get_document_repository] = lambda: repo_mock
    app.dependency_overrides[get_ingest_use_case] = lambda: use_case_mock
    app.dependency_overrides[get_settings] = lambda: Settings(admin_api_key="secret-key")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        res_no_key = await client.post("/knowledge/documents/doc-failed-1/retry")
        assert res_no_key.status_code == 401

        res_invalid_key = await client.post(
            "/knowledge/documents/doc-failed-1/retry",
            headers={"x-admin-key": "wrong-key"},
        )
        assert res_invalid_key.status_code == 401

        res_valid_key = await client.post(
            "/knowledge/documents/doc-failed-1/retry",
            headers={"x-admin-key": "secret-key"},
        )
        assert res_valid_key.status_code == 200


@pytest.mark.asyncio
async def test_list_and_get_documents_includes_error_message() -> None:
    repo_mock = MagicMock(spec=DocumentRepository)
    failed_doc = Document(
        id="doc-failed-1",
        file_name="bad.pdf",
        mime_type="application/pdf",
        status="failed",
        chunk_count=0,
        created_at=datetime.now(UTC),
        error_message="인제스트 차단: 시스템 프롬프트 유출 시도",
    )
    repo_mock.find_all = AsyncMock(return_value=[failed_doc])
    repo_mock.find_by_id = AsyncMock(return_value=failed_doc)

    app.dependency_overrides[get_document_repository] = lambda: repo_mock

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # GET /knowledge/documents
        list_res = await client.get("/knowledge/documents")
        assert list_res.status_code == 200
        items = list_res.json()
        assert len(items) == 1
        assert items[0]["id"] == "doc-failed-1"
        assert items[0]["errorMessage"] == "인제스트 차단: 시스템 프롬프트 유출 시도"

        # GET /knowledge/documents/{id}
        get_res = await client.get("/knowledge/documents/doc-failed-1")
        assert get_res.status_code == 200
        data = get_res.json()
        assert data["id"] == "doc-failed-1"
        assert data["errorMessage"] == "인제스트 차단: 시스템 프롬프트 유출 시도"


@pytest.mark.asyncio
async def test_retry_document_ingest_success_clears_error_message_and_updates_status() -> None:
    repo_mock = MagicMock(spec=DocumentRepository)
    failed_doc = Document(
        id="doc-retry-success",
        file_name="sample.txt",
        mime_type="text/plain",
        status="failed",
        chunk_count=0,
        created_at=datetime.now(UTC),
        error_message="임베딩 생성 타임아웃",
    )
    processed_doc = failed_doc.mark_processed(3)

    repo_mock.find_by_id = AsyncMock(return_value=failed_doc)
    repo_mock.get_original_file = AsyncMock(
        return_value=(b"sample content", "sample.txt", "text/plain")
    )
    repo_mock.find_all = AsyncMock(return_value=[processed_doc])

    use_case_mock = MagicMock(spec=IngestDocumentUseCase)
    use_case_mock.execute = AsyncMock(return_value=processed_doc)

    app.dependency_overrides[get_document_repository] = lambda: repo_mock
    app.dependency_overrides[get_ingest_use_case] = lambda: use_case_mock

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # POST /knowledge/documents/{id}/retry
        retry_res = await client.post("/knowledge/documents/doc-retry-success/retry")
        assert retry_res.status_code == 200
        retry_data = retry_res.json()
        assert retry_data["id"] == "doc-retry-success"
        assert retry_data["status"] == "processed"
        assert retry_data["chunkCount"] == 3
        assert retry_data["errorMessage"] is None

        # GET /knowledge/documents
        list_res = await client.get("/knowledge/documents")
        assert list_res.status_code == 200
        items = list_res.json()
        assert len(items) == 1
        assert items[0]["id"] == "doc-retry-success"
        assert items[0]["status"] == "processed"
        assert items[0]["errorMessage"] is None


@pytest.mark.asyncio
async def test_list_and_get_legacy_failed_documents_without_error_message() -> None:
    repo_mock = MagicMock(spec=DocumentRepository)
    legacy_failed_doc = Document(
        id="doc-legacy-1",
        file_name="legacy.txt",
        mime_type="text/plain",
        status="failed",
        chunk_count=0,
        created_at=datetime.now(UTC),
        error_message=None,
    )
    repo_mock.find_all = AsyncMock(return_value=[legacy_failed_doc])
    repo_mock.find_by_id = AsyncMock(return_value=legacy_failed_doc)

    app.dependency_overrides[get_document_repository] = lambda: repo_mock

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # GET /knowledge/documents
        list_res = await client.get("/knowledge/documents")
        assert list_res.status_code == 200
        items = list_res.json()
        assert len(items) == 1
        assert items[0]["id"] == "doc-legacy-1"
        assert items[0]["status"] == "failed"
        assert items[0]["errorMessage"] is None

        # GET /knowledge/documents/{id}
        get_res = await client.get("/knowledge/documents/doc-legacy-1")
        assert get_res.status_code == 200
        data = get_res.json()
        assert data["id"] == "doc-legacy-1"
        assert data["status"] == "failed"
        assert data["errorMessage"] is None
