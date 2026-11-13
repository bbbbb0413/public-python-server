from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from bson import ObjectId
from gridfs.errors import NoFile

from ai_service.knowledge.repository import DocumentRepository
from ai_service.knowledge.schemas import Document


def _make_repo() -> tuple[DocumentRepository, MagicMock]:
    # motor의 AsyncIOMotorGridFSBucket 생성자는 진짜 AsyncIOMotorDatabase 타입을
    # 요구해 MagicMock으로 대체할 수 없다 — __init__을 건너뛰고 필드를 직접 채운다.
    repo = DocumentRepository.__new__(DocumentRepository)
    repo._collection = MagicMock()
    mock_gridfs = MagicMock()
    repo._gridfs = mock_gridfs
    return repo, mock_gridfs


def _empty_async_cursor():
    async def _gen():
        return
        yield  # pragma: no cover

    return _gen()


@pytest.mark.asyncio
async def test_save_original_file_deletes_existing_then_uploads():
    repo, mock_gridfs = _make_repo()
    mock_gridfs.find = MagicMock(return_value=_empty_async_cursor())
    mock_gridfs.upload_from_stream = AsyncMock()

    await repo.save_original_file("doc-1", b"hello world", "a.txt", "text/plain")

    mock_gridfs.find.assert_called_once_with({"filename": "doc-1"})
    mock_gridfs.upload_from_stream.assert_awaited_once_with(
        "doc-1",
        b"hello world",
        metadata={"fileName": "a.txt", "mimeType": "text/plain"},
    )


@pytest.mark.asyncio
async def test_save_original_file_removes_previous_version_first():
    repo, mock_gridfs = _make_repo()

    async def existing_cursor():
        old = MagicMock()
        old._id = "old-file-id"
        yield old

    mock_gridfs.find = MagicMock(return_value=existing_cursor())
    mock_gridfs.delete = AsyncMock()
    mock_gridfs.upload_from_stream = AsyncMock()

    await repo.save_original_file("doc-1", b"new content", "a.txt", "text/plain")

    mock_gridfs.delete.assert_awaited_once_with("old-file-id")
    mock_gridfs.upload_from_stream.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_original_file_returns_content_and_metadata():
    repo, mock_gridfs = _make_repo()

    stream = MagicMock()
    stream.read = AsyncMock(return_value=b"hello world")
    stream.metadata = {"fileName": "a.txt", "mimeType": "text/plain"}
    mock_gridfs.open_download_stream_by_name = AsyncMock(return_value=stream)

    result = await repo.get_original_file("doc-1")

    assert result == (b"hello world", "a.txt", "text/plain")
    mock_gridfs.open_download_stream_by_name.assert_awaited_once_with("doc-1")


@pytest.mark.asyncio
async def test_get_original_file_falls_back_when_metadata_missing():
    repo, mock_gridfs = _make_repo()

    stream = MagicMock()
    stream.read = AsyncMock(return_value=b"data")
    stream.metadata = None
    mock_gridfs.open_download_stream_by_name = AsyncMock(return_value=stream)

    content, file_name, mime_type = await repo.get_original_file("doc-1")

    assert content == b"data"
    assert file_name == "doc-1"
    assert mime_type == "application/octet-stream"


@pytest.mark.asyncio
async def test_get_original_file_returns_none_when_missing():
    repo, mock_gridfs = _make_repo()
    mock_gridfs.open_download_stream_by_name = AsyncMock(side_effect=NoFile())

    result = await repo.get_original_file("doc-missing")

    assert result is None


@pytest.mark.asyncio
async def test_remove_deletes_original_file_and_document_record():
    repo, mock_gridfs = _make_repo()
    mock_gridfs.find = MagicMock(return_value=_empty_async_cursor())
    repo._collection.delete_one = AsyncMock()

    await repo.remove("507f1f77bcf86cd799439011")

    mock_gridfs.find.assert_called_once_with({"filename": "507f1f77bcf86cd799439011"})
    repo._collection.delete_one.assert_awaited_once()


@pytest.mark.asyncio
async def test_find_by_id_returns_none_when_invalid_object_id() -> None:
    repo, _ = _make_repo()
    repo._collection.find_one = AsyncMock()

    result = await repo.find_by_id("doc-not-found")

    assert result is None
    repo._collection.find_one.assert_not_called()


@pytest.mark.asyncio
async def test_persist_saves_error_message():
    repo, _ = _make_repo()
    inserted_id = ObjectId()
    repo._collection.insert_one = AsyncMock(return_value=MagicMock(inserted_id=inserted_id))

    doc = Document.create("sample.txt", "text/plain").mark_failed("인제스트 실패")
    result = await repo.persist(doc)

    repo._collection.insert_one.assert_awaited_once()
    saved_record = repo._collection.insert_one.await_args[0][0]
    assert saved_record["errorMessage"] == "인제스트 실패"
    assert result.error_message == "인제스트 실패"


@pytest.mark.asyncio
async def test_update_updates_error_message():
    repo, _ = _make_repo()
    doc_id = str(ObjectId())
    repo._collection.update_one = AsyncMock()

    doc = Document(
        id=doc_id,
        file_name="sample.txt",
        mime_type="text/plain",
        status="failed",
        chunk_count=0,
        created_at=datetime.now(UTC),
        error_message="Qdrant 연결 시간 초과",
    )
    result = await repo.update(doc)

    assert result == doc
    repo._collection.update_one.assert_awaited_once_with(
        {"_id": ObjectId(doc_id)},
        {"$set": {"status": "failed", "chunkCount": 0, "errorMessage": "Qdrant 연결 시간 초과"}},
    )


def test_to_domain_maps_error_message_and_falls_back():
    obj_id = ObjectId()
    # 1. errorMessage 필드가 있는 경우
    record_with_error = {
        "_id": obj_id,
        "fileName": "a.txt",
        "mimeType": "text/plain",
        "status": "failed",
        "chunkCount": 0,
        "createdAt": datetime.now(UTC),
        "errorMessage": "파일 손상",
    }
    doc1 = DocumentRepository._to_domain(record_with_error)
    assert doc1.error_message == "파일 손상"

    # 2. 레거시 데이터(errorMessage 필드 누락)
    legacy_record = {
        "_id": obj_id,
        "fileName": "b.txt",
        "mimeType": "text/plain",
        "status": "failed",
        "chunkCount": 0,
        "createdAt": datetime.now(UTC),
    }
    doc2 = DocumentRepository._to_domain(legacy_record)
    assert doc2.error_message is None
