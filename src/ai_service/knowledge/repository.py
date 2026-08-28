import logging
import uuid
from datetime import UTC, datetime
from typing import Any

from bson import ObjectId
from gridfs.errors import NoFile
from motor.motor_asyncio import AsyncIOMotorDatabase, AsyncIOMotorGridFSBucket
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchAny,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from ai_service.knowledge.schemas import (
    Document,
    DocumentProps,
    SimilaritySearchResult,
    VectorDocument,
    VectorDocumentMetadata,
)

logger = logging.getLogger(__name__)

COLLECTION_NAME = "knowledge_documents"
VECTOR_COLLECTION_NAME = "knowledge_chunks"

PARENT_LOOKUP_SCORE = 0.5
DOCUMENT_LOOKUP_SCORE = 1.0
_SCROLL_PAGE_LIMIT = 10_000


class DocumentRepository:
    def __init__(self, db: AsyncIOMotorDatabase[dict[str, Any]]) -> None:
        self._collection = db[COLLECTION_NAME]
        self._gridfs = AsyncIOMotorGridFSBucket(db, bucket_name="knowledge_files")

    async def save_original_file(
        self, document_id: str, content: bytes, file_name: str, mime_type: str
    ) -> None:
        """업로드된 원본 파일을 GridFS에 저장한다.

        재인제스트 시 기존 파일을 지우고 새로 저장한다.
        """
        await self._delete_original_file(document_id)
        await self._gridfs.upload_from_stream(
            document_id,
            content,
            metadata={"fileName": file_name, "mimeType": mime_type},
        )

    async def get_original_file(self, document_id: str) -> tuple[bytes, str, str] | None:
        """(content, file_name, mime_type)를 반환한다. 원본이 없으면 None."""
        try:
            stream = await self._gridfs.open_download_stream_by_name(document_id)
        except NoFile:
            return None
        content: bytes = await stream.read()
        metadata = stream.metadata or {}
        return (
            content,
            metadata.get("fileName", document_id),
            metadata.get("mimeType", "application/octet-stream"),
        )

    async def _delete_original_file(self, document_id: str) -> None:
        async for grid_out in self._gridfs.find({"filename": document_id}):
            await self._gridfs.delete(grid_out._id)

    async def persist(self, document: Document) -> Document:
        record: dict[str, Any] = {
            "fileName": document.file_name,
            "mimeType": document.mime_type,
            "status": document.status,
            "chunkCount": document.chunk_count,
            "createdAt": document.created_at,
        }
        result = await self._collection.insert_one(record)
        return self._to_domain({**record, "_id": result.inserted_id})

    async def find_by_id(self, id: str) -> Document | None:
        record = await self._collection.find_one({"_id": ObjectId(id)})
        return self._to_domain(record) if record else None

    async def find_all(self) -> list[Document]:
        cursor = self._collection.find().sort("createdAt", -1)
        records = await cursor.to_list(length=None)
        return [self._to_domain(r) for r in records]

    async def update(self, document: Document) -> Document:
        await self._collection.update_one(
            {"_id": ObjectId(document.id)},
            {"$set": {"status": document.status, "chunkCount": document.chunk_count}},
        )
        return document

    async def remove(self, id: str) -> None:
        await self._delete_original_file(id)
        await self._collection.delete_one({"_id": ObjectId(id)})

    @staticmethod
    def _to_domain(record: dict[str, Any]) -> Document:
        return Document.restore(
            DocumentProps(
                id=str(record["_id"]) if record.get("_id") is not None else None,
                file_name=record["fileName"],
                mime_type=record["mimeType"],
                status=record["status"],
                chunk_count=record["chunkCount"],
                created_at=record.get("createdAt") or datetime.now(UTC),
            )
        )


def _to_point_id(raw_id: str) -> str:
    return str(uuid.UUID(hex=raw_id[:32]))


def _to_metadata(payload: dict[str, object]) -> VectorDocumentMetadata:
    return VectorDocumentMetadata(
        document_id=str(payload["documentId"]),
        file_name=str(payload["fileName"]),
        chunk_index=int(payload["chunkIndex"]),  # type: ignore[call-overload]
        char_count=payload.get("charCount"),  # type: ignore[arg-type]
        parent_text=payload.get("parentText"),  # type: ignore[arg-type]
        parent_chunk_id=payload.get("parentChunkId"),  # type: ignore[arg-type]
    )


class QdrantVectorAdapter:
    def __init__(
        self,
        client: AsyncQdrantClient,
        vector_size: int,
        collection_name: str = VECTOR_COLLECTION_NAME,
    ) -> None:
        self._client = client
        self._vector_size = vector_size
        self._collection_name = collection_name

    async def ensure_collection(self) -> None:
        exists = await self._client.collection_exists(self._collection_name)
        if not exists:
            await self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=VectorParams(size=self._vector_size, distance=Distance.COSINE),
            )
            logger.info(
                "Qdrant 컬렉션 생성 완료: %s (%d차원)", self._collection_name, self._vector_size
            )

        for field_name in ("documentId", "parentChunkId"):
            try:
                await self._client.create_payload_index(
                    self._collection_name,
                    field_name=field_name,
                    field_schema=PayloadSchemaType.KEYWORD,
                )
            except Exception as e:  # noqa: BLE001
                logger.debug("payload 인덱스(%s) 생성 건너뜀: %s", field_name, e)

        try:
            await self._client.create_payload_index(
                self._collection_name,
                field_name="text",
                field_schema=PayloadSchemaType.TEXT,
            )
        except Exception as e:  # noqa: BLE001
            logger.debug("payload 인덱스(text) 생성 건너뜀: %s", e)

    async def upsert(self, documents: list[VectorDocument]) -> None:
        if not documents:
            return

        points = [
            PointStruct(
                id=_to_point_id(doc.id),
                vector=doc.embedding,
                payload={
                    "id": doc.id,
                    "text": doc.text,
                    "documentId": doc.metadata.document_id,
                    "fileName": doc.metadata.file_name,
                    "chunkIndex": doc.metadata.chunk_index,
                    **(
                        {"charCount": doc.metadata.char_count}
                        if doc.metadata.char_count is not None
                        else {}
                    ),
                    **(
                        {"parentText": doc.metadata.parent_text}
                        if doc.metadata.parent_text is not None
                        else {}
                    ),
                    **(
                        {"parentChunkId": doc.metadata.parent_chunk_id}
                        if doc.metadata.parent_chunk_id is not None
                        else {}
                    ),
                },
            )
            for doc in documents
        ]
        await self._client.upsert(collection_name=self._collection_name, points=points)

    async def similarity_search(
        self, query_embedding: list[float], top_k: int
    ) -> list[SimilaritySearchResult]:
        response = await self._client.query_points(
            collection_name=self._collection_name,
            query=query_embedding,
            limit=top_k,
            with_payload=True,
        )
        return [
            SimilaritySearchResult(
                text=str(point.payload["text"]) if point.payload else "",
                score=point.score,
                metadata=_to_metadata(point.payload or {}),
            )
            for point in response.points
        ]

    async def find_by_parent_chunk_ids(
        self, parent_chunk_ids: list[str]
    ) -> list[SimilaritySearchResult]:
        if not parent_chunk_ids:
            return []

        records, _ = await self._client.scroll(
            collection_name=self._collection_name,
            scroll_filter=Filter(
                must=[FieldCondition(key="parentChunkId", match=MatchAny(any=parent_chunk_ids))]
            ),
            limit=_SCROLL_PAGE_LIMIT,
            with_payload=True,
        )
        results = [
            SimilaritySearchResult(
                text=str(r.payload["text"]) if r.payload else "",
                score=PARENT_LOOKUP_SCORE,
                metadata=_to_metadata(r.payload or {}),
            )
            for r in records
        ]
        results.sort(key=lambda r: r.metadata.chunk_index)
        return results

    async def find_chunks_by_document_id(self, document_id: str) -> list[SimilaritySearchResult]:
        records, _ = await self._client.scroll(
            collection_name=self._collection_name,
            scroll_filter=Filter(
                must=[FieldCondition(key="documentId", match=MatchValue(value=document_id))]
            ),
            limit=_SCROLL_PAGE_LIMIT,
            with_payload=True,
        )
        results = [
            SimilaritySearchResult(
                text=str(r.payload["text"]) if r.payload else "",
                score=DOCUMENT_LOOKUP_SCORE,
                metadata=_to_metadata(r.payload or {}),
            )
            for r in records
        ]
        results.sort(key=lambda r: r.metadata.chunk_index)
        return results

    async def delete_by_document_id(self, document_id: str) -> None:
        await self._client.delete(
            collection_name=self._collection_name,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[FieldCondition(key="documentId", match=MatchValue(value=document_id))]
                )
            ),
        )


# Backward compatibility aliases
DocumentRepositoryImpl = DocumentRepository
QdrantVectorStore = QdrantVectorAdapter

__all__ = [
    "COLLECTION_NAME",
    "DOCUMENT_LOOKUP_SCORE",
    "DocumentRepository",
    "DocumentRepositoryImpl",
    "PARENT_LOOKUP_SCORE",
    "QdrantVectorAdapter",
    "QdrantVectorStore",
    "VECTOR_COLLECTION_NAME",
]
