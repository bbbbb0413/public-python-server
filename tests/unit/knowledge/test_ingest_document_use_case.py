from unittest.mock import AsyncMock, MagicMock

import pytest

from ai_service.knowledge.application.ingest_document_use_case import IngestDocumentUseCase
from ai_service.knowledge.schemas import Document, IngestDocumentCommand
from ai_service.rag.schemas import GuardrailVerdict


@pytest.fixture
def mock_dependencies():
    doc_repo = MagicMock()
    doc_repo.find_by_id = AsyncMock(return_value=None)
    doc_repo.persist = AsyncMock(
        side_effect=lambda doc: Document(
            id="doc-123",
            file_name=doc.file_name,
            mime_type=doc.mime_type,
            chunk_count=doc.chunk_count,
            status=doc.status,
            created_at=doc.created_at,
        )
    )
    doc_repo.update = AsyncMock(side_effect=lambda doc: doc)
    doc_repo.save_original_file = AsyncMock()

    vector_store = MagicMock()
    vector_store.delete_by_document_id = AsyncMock()
    vector_store.upsert = AsyncMock()

    embedding_provider = MagicMock()
    embedding_provider.embed = AsyncMock(return_value=[[0.1, 0.2, 0.3]])

    llm_provider = MagicMock()

    rag_validator = MagicMock()
    rag_validator.inspect_input = MagicMock(return_value=GuardrailVerdict.allow())

    return {
        "doc_repo": doc_repo,
        "vector_store": vector_store,
        "embedding_provider": embedding_provider,
        "llm_provider": llm_provider,
        "rag_validator": rag_validator,
    }


@pytest.mark.asyncio
async def test_ingest_document_use_case_publishes_progress_steps(mock_dependencies):
    use_case = IngestDocumentUseCase(
        document_repo=mock_dependencies["doc_repo"],
        vector_store=mock_dependencies["vector_store"],
        embedding_provider=mock_dependencies["embedding_provider"],
        llm_provider=mock_dependencies["llm_provider"],
        rag_validator=mock_dependencies["rag_validator"],
        contextual_embeddings_enabled=False,
    )

    progress_events: list[tuple[str, int]] = []

    async def on_progress(step: str, progress: int) -> None:
        progress_events.append((step, progress))

    command = IngestDocumentCommand(
        file_name="test.txt",
        mime_type="text/plain",
        content=b"Sample document content for testing ingestion.",
    )

    result = await use_case.execute(command, on_progress=on_progress)

    assert result.status == "processed"
    assert progress_events == [
        ("extract", 20),
        ("chunk", 45),
        ("embed", 70),
        ("index", 90),
    ]


@pytest.mark.asyncio
async def test_ingest_document_use_case_without_on_progress(mock_dependencies):
    use_case = IngestDocumentUseCase(
        document_repo=mock_dependencies["doc_repo"],
        vector_store=mock_dependencies["vector_store"],
        embedding_provider=mock_dependencies["embedding_provider"],
        llm_provider=mock_dependencies["llm_provider"],
        rag_validator=mock_dependencies["rag_validator"],
        contextual_embeddings_enabled=False,
    )

    command = IngestDocumentCommand(
        file_name="test.txt",
        mime_type="text/plain",
        content=b"Sample document content for testing ingestion.",
    )

    result = await use_case.execute(command)

    assert result.status == "processed"
