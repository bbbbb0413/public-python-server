import logging
import uuid

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

from ai_service.knowledge.domain.port.vector_store_port import (
    SimilaritySearchResult,
    VectorDocument,
    VectorDocumentMetadata,
)

logger = logging.getLogger(__name__)

COLLECTION_NAME = "knowledge_chunks"

# 부모 청크 조회(findByParentChunkIds)는 벡터 검색이 아니므로 유사도 점수 대신
# NestJS 구현(mongodb-vector.adapter.ts)과 동일하게 중립값 0.5를 사용한다.
PARENT_LOOKUP_SCORE = 0.5
# 문서 단위 전체 조회(findChunksByDocumentId)도 유사도 개념이 없으므로 1.0(완전 일치)을 사용한다.
DOCUMENT_LOOKUP_SCORE = 1.0

_SCROLL_PAGE_LIMIT = 10_000


def _to_point_id(raw_id: str) -> str:
    """Qdrant point ID는 unsigned int 또는 UUID만 허용한다.

    knowledge 도메인의 id는 SHA-256 해시(hex 64자)이므로, 앞 32자를 이용해
    결정적(deterministic) UUID를 생성한다. 원본 id는 payload["id"]에 그대로 보존한다.
    """
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
        collection_name: str = COLLECTION_NAME,
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
            except Exception as e:  # noqa: BLE001 - 인덱스가 이미 있으면 무시
                logger.debug("payload 인덱스(%s) 생성 건너뜀: %s", field_name, e)

        # rag 모듈의 QdrantTextSearchAdapter(렉시컬 검색)가 MatchText 필터로 사용한다.
        try:
            await self._client.create_payload_index(
                self._collection_name,
                field_name="text",
                field_schema=PayloadSchemaType.TEXT,
            )
        except Exception as e:  # noqa: BLE001 - 인덱스가 이미 있으면 무시
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
