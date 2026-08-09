import re
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ai_service.prompt.domain.model.prompt_template import PromptTemplate

_NAME_PATTERN = re.compile(r"^[a-z0-9-]+$")


class CreatePromptIn(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    name: str = Field(description="프롬프트 이름 (소문자·숫자·하이픈)", examples=["rag-qa-system"])
    content: str = Field(description="프롬프트 본문. {{context}} 같은 변수 사용 가능")
    variables: list[str] = Field(description="치환 변수 목록", examples=[["context"]])
    user_id: str | None = Field(
        default=None, alias="userId", description="사용자별 프롬프트일 경우 해당 userId"
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        if not value or not _NAME_PATTERN.match(value):
            raise ValueError("이름은 소문자, 숫자, 하이픈만 허용됩니다.")
        return value


class PromptOut(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str | None
    name: str
    version: int
    content: str
    is_active: bool = Field(alias="isActive")
    variables: list[str]
    user_id: str | None = Field(default=None, alias="userId")
    created_at: datetime = Field(alias="createdAt")
    updated_at: datetime = Field(alias="updatedAt")

    @classmethod
    def from_domain(cls, template: PromptTemplate) -> "PromptOut":
        return cls(
            id=template.id,
            name=template.name.get_value(),
            version=template.version,
            content=template.content,
            isActive=template.is_active,
            variables=template.variables,
            userId=template.user_id,
            createdAt=template.created_at,
            updatedAt=template.updated_at,
        )
