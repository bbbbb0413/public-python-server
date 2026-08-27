import logging

import httpx

from ai_service.knowledge.schemas import SimilaritySearchResult

logger = logging.getLogger(__name__)


class HttpRerankerAdapter:
    def __init__(self, api_url: str | None, api_key: str | None) -> None:
        self._api_url = api_url
        self._api_key = api_key

    async def rerank(
        self, query: str, chunks: list[SimilaritySearchResult], top_n: int
    ) -> list[SimilaritySearchResult]:
        if not self._api_url:
            logger.warning("RERANKER_API_URL이 설정되지 않아 리랭킹을 건너뜁니다.")
            return chunks[:top_n]

        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self._api_url,
                    headers=headers,
                    json={
                        "query": query,
                        "documents": [c.metadata.parent_text or c.text for c in chunks],
                        "top_n": top_n,
                    },
                )

            if response.status_code >= 400:
                logger.warning(
                    "Reranker API 오류 (%d), 원본 순서로 반환합니다.", response.status_code
                )
                return chunks[:top_n]

            data = response.json()
            results = sorted(
                data["results"], key=lambda item: item["relevance_score"], reverse=True
            )
            return [
                SimilaritySearchResult(
                    text=chunks[item["index"]].text,
                    score=item["relevance_score"],
                    metadata=chunks[item["index"]].metadata,
                )
                for item in results
            ]
        except Exception as e:
            logger.warning("Reranker API 호출 실패: %s, 원본 순서로 반환합니다.", e)
            return chunks[:top_n]
