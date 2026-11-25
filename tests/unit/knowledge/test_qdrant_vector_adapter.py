from unittest.mock import AsyncMock, MagicMock

import pytest
from qdrant_client.models import FieldCondition, Filter, MatchValue, PointIdsList

from ai_service.knowledge.repository import QdrantVectorAdapter


@pytest.fixture
def mock_qdrant_client():
    return MagicMock()


@pytest.mark.asyncio
async def test_find_point_ids_by_document_id(mock_qdrant_client):
    adapter = QdrantVectorAdapter(client=mock_qdrant_client, vector_size=1536)

    record1 = MagicMock()
    record1.id = "uuid-point-1"
    record2 = MagicMock()
    record2.id = "uuid-point-2"

    mock_qdrant_client.scroll = AsyncMock(return_value=([record1, record2], None))

    point_ids = await adapter.find_point_ids_by_document_id("doc-123")

    assert point_ids == ["uuid-point-1", "uuid-point-2"]
    mock_qdrant_client.scroll.assert_awaited_once_with(
        collection_name=adapter._collection_name,
        scroll_filter=Filter(
            must=[FieldCondition(key="documentId", match=MatchValue(value="doc-123"))]
        ),
        limit=10_000,
        with_payload=False,
        with_vectors=False,
    )


@pytest.mark.asyncio
async def test_delete_points_by_ids_with_points(mock_qdrant_client):
    adapter = QdrantVectorAdapter(client=mock_qdrant_client, vector_size=1536)
    mock_qdrant_client.delete = AsyncMock()

    await adapter.delete_points_by_ids(["point-1", "point-2"])

    mock_qdrant_client.delete.assert_awaited_once_with(
        collection_name=adapter._collection_name,
        points_selector=PointIdsList(points=["point-1", "point-2"]),
    )


@pytest.mark.asyncio
async def test_delete_points_by_ids_with_empty_list(mock_qdrant_client):
    adapter = QdrantVectorAdapter(client=mock_qdrant_client, vector_size=1536)
    mock_qdrant_client.delete = AsyncMock()

    await adapter.delete_points_by_ids([])

    mock_qdrant_client.delete.assert_not_called()
