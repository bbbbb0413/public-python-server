from qdrant_client import AsyncQdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchText

from ai_service.knowledge.schemas import (
    SimilaritySearchResult,
    VectorDocumentMetadata,
)

COLLECTION_NAME = "knowledge_chunks"

# TS(MongoTextSearchAdapter)는 MongoDB $text의 textScore로 실제 관련도 순위를 매겼지만,
# Qdrant의 MatchText 필터는 매칭 여부만 반환하고 관련도 점수를 주지 않는다.
# RRF 융합은 리스트 내 "순위(index)"만 사용하므로 동작은 하지만, 렉시컬 랭킹 품질은
# TS 버전보다 낮다 — 이후 Qdrant sparse vector(BM25) 도입을 후속 과제로 남긴다.
_MATCH_SCORE = 0.5


class QdrantTextSearchAdapter:
    def __init__(self, client: AsyncQdrantClient, collection_name: str = COLLECTION_NAME) -> None:
        self._client = client
        self._collection_name = collection_name

    async def search(self, query: str, top_k: int) -> list[SimilaritySearchResult]:
        records, _ = await self._client.scroll(
            collection_name=self._collection_name,
            scroll_filter=Filter(must=[FieldCondition(key="text", match=MatchText(text=query))]),
            limit=top_k,
            with_payload=True,
        )

        return [
            SimilaritySearchResult(
                text=str(r.payload["text"]) if r.payload else "",
                score=_MATCH_SCORE,
                metadata=self._to_metadata(r.payload or {}),
            )
            for r in records
        ]

    @staticmethod
    def _to_metadata(payload: dict[str, object]) -> VectorDocumentMetadata:
        return VectorDocumentMetadata(
            document_id=str(payload["documentId"]),
            file_name=str(payload["fileName"]),
            chunk_index=int(payload["chunkIndex"]),  # type: ignore[call-overload]
            char_count=payload.get("charCount"),  # type: ignore[arg-type]
            parent_text=payload.get("parentText"),  # type: ignore[arg-type]
            parent_chunk_id=payload.get("parentChunkId"),  # type: ignore[arg-type]
        )
