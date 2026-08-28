from ai_service.knowledge.schemas import SimilaritySearchResult

DEFAULT_K = 60


class RrfFusionService:
    """여러 검색 결과 목록을 RRF(Reciprocal Rank Fusion)로 순위 융합한다.

    각 목록 내 순위(index)만으로 정렬 순서를 정하되, 사용자에게 노출되는
    score 필드는 RRF 합산값(항상 1/60 안팎의 작은 값이라 "관련도 %"로
    보여주면 사실상 고정값처럼 보인다) 대신 원본 검색 점수(벡터 코사인
    유사도 등) 중 가장 높은 값을 그대로 보존한다.
    """

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
                    existing_rrf, existing_doc = existing
                    best_doc = doc if doc.score > existing_doc.score else existing_doc
                    score_map[key] = (existing_rrf + rrf_score, best_doc)
                else:
                    score_map[key] = (rrf_score, doc)

        ranked = sorted(score_map.values(), key=lambda item: item[0], reverse=True)
        return [
            SimilaritySearchResult(text=doc.text, score=doc.score, metadata=doc.metadata)
            for _, doc in ranked
        ]
