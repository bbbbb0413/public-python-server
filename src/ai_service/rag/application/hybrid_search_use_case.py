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
        # Step 1: 메인 쿼리 임베딩 + (요청 시) HyDE 가설 문서 생성 + (복합 질문일 때만)
        # 쿼리 분해를 동시에 시작한다. HyDE·쿼리 분해는 각각 LLM 호출 1회를 태우므로
        # 불필요할 때 건너뛰는 것만으로 지연시간을 크게 줄일 수 있다 — command.use_hyde가
        # false인데도 매번 HyDE를 태우거나, 복합 질문이 아닌데도 매번 분해를 태우던
        # 버그(리팩터링 중 게이트가 빠짐)를 함께 고친다.
        should_decompose = self._query_decomposer.should_decompose(command.query)

        embedding_task: asyncio.Task[list[list[float]]] = asyncio.ensure_future(
            self._embedding_provider.embed([command.query])
        )
        hyde_task: asyncio.Task[str | None] | None = (
            asyncio.ensure_future(self._hyde_service.generate_hypothetical_document(command.query))
            if command.use_hyde
            else None
        )
        decompose_task: asyncio.Task[list[str]] | None = (
            asyncio.ensure_future(self._query_decomposer.decompose(command.query))
            if should_decompose
            else None
        )

        query_embedding: list[float] = (await embedding_task)[0]
        hypothetical_doc: str | None = await hyde_task if hyde_task is not None else None
        sub_queries: list[str] = await decompose_task if decompose_task is not None else []

        hyde_embedding: list[float] | None = None
        if hypothetical_doc is not None:
            hyde_embeddings = await self._embedding_provider.embed([hypothetical_doc])
            hyde_embedding = hyde_embeddings[0]

        # Step 2: 멀티 쿼리 병렬 검색 (Vector + Lexical)
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

        # Step 3: RRF(Reciprocal Rank Fusion) 융합
        fused = self._rrf_fusion.fuse(search_lists, self._rrf_k)
        candidates = fused[: self._candidate_k]

        # Step 4: 리랭커 적용 (활성화 시)
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
