from ai_service.knowledge.domain.port.vector_store_port import SimilaritySearchResult

DEFAULT_K = 60


class RrfFusionService:
    def fuse(
        self, result_lists: list[list[SimilaritySearchResult]], k: int = DEFAULT_K
    ) -> list[SimilaritySearchResult]:
        score_map: dict[str, tuple[float, SimilaritySearchResult]] = {}

        for result_list in result_lists:
            for index, doc in enumerate(result_list):
                key = f"{doc.metadata.document_id}:{doc.metadata.chunk_index}"
                rrf_score = 1 / (k + index + 1)
                existing = score_map.get(key)
                if existing is not None:
                    score_map[key] = (existing[0] + rrf_score, doc)
                else:
                    score_map[key] = (rrf_score, doc)

        ranked = sorted(score_map.values(), key=lambda item: item[0], reverse=True)
        return [
            SimilaritySearchResult(text=doc.text, score=score, metadata=doc.metadata)
            for score, doc in ranked
        ]
