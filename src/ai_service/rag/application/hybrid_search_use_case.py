import asyncio
from dataclasses import dataclass

from ai_service.config.settings import Settings
from ai_service.knowledge.domain.port.embedding_provider_port import IEmbeddingProvider
from ai_service.knowledge.domain.port.vector_store_port import (
    IVectorStorePort,
    SimilaritySearchResult,
)
from ai_service.rag.application.command.hybrid_search_command import HybridSearchCommand
from ai_service.rag.application.hyde_service import HydeService
from ai_service.rag.application.query_decomposer_service import QueryDecomposer
from ai_service.rag.application.rrf_fusion_service import RrfFusionService
from ai_service.rag.domain.port.lexical_search_port import ILexicalSearchPort
from ai_service.rag.domain.port.reranker_port import IRerankerPort

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
        embedding_provider: IEmbeddingProvider,
        vector_store: IVectorStorePort,
        lexical_search: ILexicalSearchPort,
        reranker: IRerankerPort,
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
        if self._query_decomposer.should_decompose(command.question):
            return await self._execute_decomposed(command)
        return await self._execute_simple(command)

    async def _execute_simple(self, command: HybridSearchCommand) -> HybridSearchResult:
        query_embedding = await self._resolve_query_embedding(command.question, command.use_hyde)

        dense_results, lexical_results = await asyncio.gather(
            self._vector_store.similarity_search(query_embedding, self._candidate_k),
            self._lexical_search.search(command.question, self._candidate_k),
        )

        merged = self._rrf_fusion.fuse([dense_results, lexical_results], self._rrf_k)

        ranked = merged
        if self._reranker_enabled:
            ranked = await self._reranker.rerank(command.question, merged, self._reranker_top_n)

        top_ranked = ranked[: command.top_k]
        enriched = await self._expand_with_siblings(top_ranked)

        return HybridSearchResult(query_embedding=query_embedding, chunks=enriched)

    async def _expand_with_siblings(
        self, chunks: list[SimilaritySearchResult]
    ) -> list[SimilaritySearchResult]:
        parent_chunk_ids = list(
            {c.metadata.parent_chunk_id for c in chunks if c.metadata.parent_chunk_id}
        )
        if not parent_chunk_ids:
            return chunks

        hit_indices: dict[str, int] = {}
        for c in chunks:
            if c.metadata.parent_chunk_id:
                hit_indices[c.metadata.parent_chunk_id] = c.metadata.chunk_index

        siblings = await self._vector_store.find_by_parent_chunk_ids(parent_chunk_ids)

        existing_keys = {f"{c.metadata.document_id}:{c.metadata.chunk_index}" for c in chunks}

        def is_new_sibling(s: SimilaritySearchResult) -> bool:
            if f"{s.metadata.document_id}:{s.metadata.chunk_index}" in existing_keys:
                return False
            parent_id = s.metadata.parent_chunk_id
            if not parent_id or parent_id not in hit_indices:
                return False
            return abs(s.metadata.chunk_index - hit_indices[parent_id]) <= 1

        new_siblings = [s for s in siblings if is_new_sibling(s)]
        return [*chunks, *new_siblings]

    async def _execute_decomposed(self, command: HybridSearchCommand) -> HybridSearchResult:
        sub_queries = await self._query_decomposer.decompose(command.question)

        sub_results = await asyncio.gather(
            *[
                self._execute_simple(HybridSearchCommand(sq, command.top_k, command.use_hyde))
                for sq in sub_queries
            ]
        )

        merged = self._rrf_fusion.fuse([r.chunks for r in sub_results], self._rrf_k)

        return HybridSearchResult(
            query_embedding=sub_results[0].query_embedding,
            chunks=merged[: command.top_k],
        )

    async def _resolve_query_embedding(self, question: str, use_hyde: bool) -> list[float]:
        [original_embedding] = await self._embedding_provider.embed([question])

        if not use_hyde or not self._hyde_service.should_apply(question):
            return original_embedding

        hypothetical = await self._hyde_service.generate_hypothetical(question)
        [hyde_embedding] = await self._embedding_provider.embed([hypothetical])

        similarity = self._cosine_similarity(original_embedding, hyde_embedding)
        if similarity >= HYDE_SIMILARITY_THRESHOLD:
            return [(a + b) / 2 for a, b in zip(original_embedding, hyde_embedding, strict=True)]
        return original_embedding

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b, strict=True))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(y * y for y in b) ** 0.5
        denom = norm_a * norm_b
        return 0.0 if denom == 0 else dot / denom
