from typing import Any

from ai_service.prompt.repository import PromptTemplateRepository
from ai_service.prompt.schemas import (
    CreatePromptIn,
    PromptTemplate,
    PromptTemplateNotFoundError,
    PromptTemplateProps,
)

RAG_QA_DEFAULT_PROMPT = """당신은 주어진 문서를 기반으로 질문에 정확하게 답변하는 AI 어시스턴트입니다.
오늘 날짜: {{currentDate}}

## 출력 형식 규칙 (반드시 준수)
- 오직 한국어와 마크다운(제목, 불릿, 번호 목록, 표)만 사용하세요.
- "Step 1:", "Step 2:", "## Step N" 형식을 절대 사용하지 마세요.
- "The final answer is", "In conclusion", "To summarize" 같은 영어 마무리 표현을 절대 사용하지 마세요.
- 수학 기호($), LaTeX(\\boxed{}, \\frac{} 등)를 절대 사용하지 마세요.
- 분석 과정을 단계별로 나열하지 말고, 결론을 직접 서술하세요.

## 답변 원칙
1. 반드시 아래 [컨텍스트]에 포함된 정보만을 사용하여 답변하세요.
2. 답변 시 "[출처 N]"을 인용하여 근거를 제시하세요 (예: "[출처 1]에 따르면 ...").
3. 컨텍스트에서 답변 가능한 내용은 최대한 구체적이고 상세하게 설명하세요.
4. 부분적으로만 답변 가능한 경우, 확인된 내용을 먼저 답변한 후 부족한 부분을 명시하세요.
5. 컨텍스트에 전혀 관련 정보가 없는 경우에만 "해당 정보는 제공된 문서에 포함되어 있지 않습니다."라고 답변하세요.
6. 시간적 표현(최근, 현재, 최신 등)이 포함된 질문은 오늘 날짜({{currentDate}})를 기준으로 컨텍스트의 날짜 정보를 비교하여 가장 최근 항목을 식별하세요.
7. 목록, 단계, 비교가 필요한 경우 마크다운 형식(불릿, 번호 목록, 표)을 활용하여 구조화하세요.

## 컨텍스트
{{context}}"""


class PromptService:
    def __init__(self, repo: PromptTemplateRepository | Any) -> None:
        self._repo: PromptTemplateRepository = repo

    async def get_active_prompt(self, name: str, user_id: str | None = None) -> PromptTemplate:
        if user_id:
            user_prompt = await self._repo.find_active_for_user(name, user_id)
            if user_prompt is not None:
                return user_prompt

        active = await self._repo.find_active(name)
        if active is not None:
            return active

        return PromptTemplate.restore(
            PromptTemplateProps(
                name=name,
                version=0,
                content=RAG_QA_DEFAULT_PROMPT,
                is_active=False,
                variables=["context", "currentDate"],
            )
        )

    execute = get_active_prompt

    async def create_prompt(self, dto: CreatePromptIn) -> PromptTemplate:
        existing = await self._repo.find_all_by_name(dto.name, user_id=dto.user_id)
        next_version = max((t.version for t in existing), default=0) + 1

        template = PromptTemplate.create(
            name=dto.name,
            content=dto.content,
            variables=dto.variables,
            version=next_version,
            user_id=dto.user_id,
        )
        return await self._repo.persist(template)

    async def list_versions(
        self, name: str, user_id: str | None = None
    ) -> list[PromptTemplate]:
        return await self._repo.find_all_by_name(name, user_id=user_id)

    async def activate_prompt(
        self, name: str, version: int, user_id: str | None = None
    ) -> PromptTemplate:
        target = await self._repo.find_by_name_and_version(name, version)
        if target is None or target.user_id != user_id:
            raise PromptTemplateNotFoundError(name, version)

        if user_id is not None:
            await self._repo.deactivate_all_by_name_for_user(name, user_id)
        else:
            await self._repo.deactivate_all_by_name(name)
        activated = target.activate()
        return await self._repo.update(activated)

    async def deactivate_active_prompt(self, name: str, user_id: str) -> None:
        await self._repo.deactivate_active_for_user(name, user_id)


__all__ = ["PromptService", "RAG_QA_DEFAULT_PROMPT"]
