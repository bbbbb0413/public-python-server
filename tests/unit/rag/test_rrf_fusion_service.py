from ai_service.knowledge.schemas import SimilaritySearchResult, VectorDocumentMetadata
from ai_service.rag.application.rrf_fusion_service import RrfFusionService


def _make_result(document_id: str, chunk_index: int, score: float) -> SimilaritySearchResult:
    return SimilaritySearchResult(
        text=f"chunk-{document_id}-{chunk_index}",
        score=score,
        metadata=VectorDocumentMetadata(
            document_id=document_id, file_name="doc.txt", chunk_index=chunk_index
        ),
    )


class TestRrfFusionService:
    def test_반환되는_score는_RRF_합산값이_아니라_원본_유사도_점수를_보존한다(self) -> None:
        service = RrfFusionService()
        vector_hit = _make_result("doc-1", 0, score=0.87)

        fused = service.fuse([[vector_hit]], k=60)

        assert fused[0].score == 0.87

    def test_여러_목록에_중복으로_등장하면_더_높은_원본_점수를_유지한다(self) -> None:
        service = RrfFusionService()
        vector_hit = _make_result("doc-1", 0, score=0.9)
        lexical_hit = _make_result("doc-1", 0, score=0.5)

        fused = service.fuse([[vector_hit], [lexical_hit]], k=60)

        assert len(fused) == 1
        assert fused[0].score == 0.9

    def test_여러_목록에서_순위가_높을수록_먼저_반환된다(self) -> None:
        service = RrfFusionService()
        doc_a = _make_result("doc-a", 0, score=0.4)
        doc_b = _make_result("doc-b", 0, score=0.95)

        # doc-b는 vector 검색 1등, doc-a는 vector 검색 2등 + lexical 검색 1등
        # → RRF 합산 순위로는 doc-a가 doc-b보다 앞서야 한다(두 목록에서 신호를 받음)
        vector_list = [doc_b, doc_a]
        lexical_list = [doc_a]

        fused = service.fuse([vector_list, lexical_list], k=60)

        assert [r.metadata.document_id for r in fused] == ["doc-a", "doc-b"]
        # 순위는 RRF 융합 기준이지만, 노출되는 score는 여전히 원본 유사도 점수다
        assert fused[0].score == 0.4
        assert fused[1].score == 0.95
