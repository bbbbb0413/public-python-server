import asyncio
from dataclasses import dataclass
from typing import Any

from ai_service.core.config import Settings
from ai_service.knowledge.schemas import SimilaritySearchResult
from ai_service.rag.application.command.hybrid_search_command import HybridSearchCommand
from ai_service.rag.application.hyde_service import HydeService
from ai_service.rag.application.query_decomposer_service import QueryDecomposer
from ai_service.rag.application.rrf_fusion_service import RrfFusionService

DEFAULT_CANDIDATE_K = 40
DEFAULT_RERANKER_TOP_N = 12
DEFAULT_RRF_K = 60
HYDE_SIMILARITY_THRESHOLD = 0.5


@dataclass(frozen=True)
class HybridSearchResult:
    query_embedding: list[float]
    chunks: list[SimilaritySearchResult]


class HybridSearchUseCase:
    def __init__(
        self,
        embedding_provider: Any,
        vector_store: Any,
        lexical_search: Any,
        reranker: Any,
        rrf_fusion: RrfFusionService,
        hyde_service: HydeService,
        query_decomposer: QueryDecomposer,
        settings: Settings,
    ) -> None:
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store
        self._lexical_search = lexical_search
        self._reranker = reranker
        self._rrf_fusion = rrf_fusion
        self._hyde_service = hyde_service
        self._query_decomposer = query_decomposer
        self._candidate_k = settings.hybrid_candidate_k or DEFAULT_CANDIDATE_K
        self._reranker_top_n = settings.reranker_top_n or DEFAULT_RERANKER_TOP_N
        self._rrf_k = settings.rrf_k or DEFAULT_RRF_K
        self._reranker_enabled = settings.reranker_enabled

    async def execute(self, command: HybridSearchCommand) -> HybridSearchResult:
        # Step 1: 메인 쿼리 임베딩 + HyDE 가설 문서 임베딩(생성 성공 시) 병렬 계산
        embedding_coros: list[Any] = [
            self._embedding_provider.embed([command.query]),
            self._hyde_service.generate_hypothetical_document(command.query),
        ]
        embedding_results = await asyncio.gather(*embedding_coros)
        query_embedding: list[float] = embedding_results[0][0]
        hypothetical_doc: str | None = embedding_results[1]

        hyde_embedding: list[float] | None = None
        if hypothetical_doc is not None:
            hyde_embeddings = await self._embedding_provider.embed([hypothetical_doc])
            hyde_embedding = hyde_embeddings[0]

        # Step 2: 쿼리 분해(복합 질문 분해)
        sub_queries = await self._query_decomposer.decompose(command.query)

        # Step 3: 멀티 쿼리 병렬 검색 (Vector + Lexical)
        search_coros: list[Any] = [
            self._vector_store.find_similar(query_embedding, self._candidate_k),
            self._lexical_search.search_by_text(command.query, self._candidate_k),
        ]

        if hyde_embedding is not None:
            search_coros.append(
                self._vector_store.find_similar(
                    hyde_embedding, self._candidate_k, HYDE_SIMILARITY_THRESHOLD
                )
            )

        for sub_q in sub_queries:
            search_coros.append(self._lexical_search.search_by_text(sub_q, self._candidate_k))

        raw_results = await asyncio.gather(*search_coros)
        search_lists: list[list[SimilaritySearchResult]] = [r for r in raw_results if r]

        # Step 4: RRF(Reciprocal Rank Fusion) 융합
        fused = self._rrf_fusion.fuse(search_lists, self._rrf_k)
        candidates = fused[: self._candidate_k]

        # Step 5: 리랭커 적용 (활성화 시)
        if self._reranker_enabled:
            reranked = await self._reranker.rerank(command.query, candidates, self._reranker_top_n)
            chunks = (
                reranked
                if reranked
                else self._deduplicate_by_document(candidates[: self._reranker_top_n])
            )
        else:
            chunks = self._deduplicate_by_document(candidates[: self._reranker_top_n])

        return HybridSearchResult(query_embedding=query_embedding, chunks=chunks)

    @staticmethod
    def _deduplicate_by_document(
        candidates: list[SimilaritySearchResult],
    ) -> list[SimilaritySearchResult]:
        seen_documents: set[str] = set()
        deduped: list[SimilaritySearchResult] = []
        for c in candidates:
            if c.metadata.document_id not in seen_documents:
                seen_documents.add(c.metadata.document_id)
                deduped.append(c)
        return deduped


__all__ = [
    "DEFAULT_CANDIDATE_K",
    "DEFAULT_RERANKER_TOP_N",
    "DEFAULT_RRF_K",
    "HybridSearchResult",
    "HybridSearchUseCase",
]
