import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

_NAME_PATTERN = re.compile(r"^[a-z0-9-]+$")


class PromptTemplateNotFoundError(Exception):
    def __init__(self, name: str, version: int) -> None:
        super().__init__(f"프롬프트 템플릿을 찾을 수 없습니다: name={name}, version={version}")
        self.name = name
        self.version = version


class PromptName:
    @staticmethod
    def validate(value: str) -> str:
        if not value or not value.strip():
            raise ValueError("프롬프트 이름은 비어있을 수 없습니다.")
        if not _NAME_PATTERN.match(value):
            raise ValueError("프롬프트 이름은 소문자, 숫자, 하이픈만 허용됩니다.")
        return value

    def __init__(self, value: str) -> None:
        self._value = self.validate(value)

    def get_value(self) -> str:
        return self._value

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, PromptName):
            return self._value == other._value
        if isinstance(other, str):
            return self._value == other
        return False

    @classmethod
    def of(cls, value: str) -> "PromptName":
        return cls(value)


@dataclass(frozen=True)
class PromptTemplateProps:
    name: str
    version: int
    content: str
    is_active: bool
    variables: list[str] = field(default_factory=list)
    id: str | None = None
    user_id: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class PromptTemplate:
    def __init__(
        self,
        id: str | None,
        name: PromptName,
        version: int,
        content: str,
        is_active: bool,
        variables: list[str],
        created_at: datetime,
        updated_at: datetime,
        user_id: str | None = None,
    ) -> None:
        self.id = id
        self.name = name
        self.version = version
        self.content = content
        self.is_active = is_active
        self.variables = variables
        self.created_at = created_at
        self.updated_at = updated_at
        self.user_id = user_id

    @classmethod
    def create(
        cls,
        name: str,
        content: str,
        variables: list[str] | None = None,
        version: int = 1,
        user_id: str | None = None,
    ) -> "PromptTemplate":
        now = datetime.now(UTC)
        return cls(
            id=None,
            name=PromptName.of(name),
            version=version,
            content=content,
            is_active=False,
            variables=variables or [],
            created_at=now,
            updated_at=now,
            user_id=user_id,
        )

    @classmethod
    def restore(cls, props: PromptTemplateProps) -> "PromptTemplate":
        now = datetime.now(UTC)
        return cls(
            id=props.id,
            name=PromptName.of(props.name),
            version=props.version,
            content=props.content,
            is_active=props.is_active,
            variables=props.variables,
            created_at=props.created_at or now,
            updated_at=props.updated_at or now,
            user_id=props.user_id,
        )

    def activate(self) -> "PromptTemplate":
        return PromptTemplate.restore(
            PromptTemplateProps(
                id=self.id,
                name=self.name.get_value(),
                version=self.version,
                content=self.content,
                is_active=True,
                variables=self.variables,
                user_id=self.user_id,
                created_at=self.created_at,
                updated_at=datetime.now(UTC),
            )
        )

    def deactivate(self) -> "PromptTemplate":
        return PromptTemplate.restore(
            PromptTemplateProps(
                id=self.id,
                name=self.name.get_value(),
                version=self.version,
                content=self.content,
                is_active=False,
                variables=self.variables,
                user_id=self.user_id,
                created_at=self.created_at,
                updated_at=datetime.now(UTC),
            )
        )

    def render(self, variables: dict[str, str]) -> str:
        result = self.content
        for key, val in variables.items():
            result = result.replace(f"{{{{{key}}}}}", val)
        return result


class CreatePromptIn(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    name: str
    content: str
    variables: list[str] = Field(default_factory=list)
    user_id: str | None = Field(default=None, alias="userId")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        return PromptName.validate(v)


class PromptOut(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str | None = None
    name: str
    version: int
    content: str
    is_active: bool = Field(alias="isActive")
    variables: list[str] = Field(default_factory=list)
    user_id: str | None = Field(default=None, alias="userId")
    created_at: datetime = Field(alias="createdAt")
    updated_at: datetime = Field(alias="updatedAt")

    def render(self, variables: dict[str, str]) -> str:
        result = self.content
        for key, val in variables.items():
            result = result.replace(f"{{{{{key}}}}}", val)
        return result

    @classmethod
    def from_domain(cls, template: Any) -> "PromptOut":
        if isinstance(template, PromptOut):
            return template
        name_val = (
            template.name.get_value()
            if hasattr(template.name, "get_value")
            else str(template.name)
        )
        return cls(
            id=template.id,
            name=name_val,
            version=template.version,
            content=template.content,
            isActive=template.is_active,
            variables=list(template.variables),
            userId=template.user_id,
            createdAt=template.created_at,
            updatedAt=template.updated_at,
        )


__all__ = [
    "CreatePromptIn",
    "PromptName",
    "PromptOut",
    "PromptTemplate",
    "PromptTemplateNotFoundError",
    "PromptTemplateProps",
]
