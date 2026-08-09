from dataclasses import dataclass, field
from datetime import UTC, datetime

from ai_service.prompt.domain.vo.prompt_name import PromptName
from ai_service.shared_kernel.aggregate_root import AggregateRoot


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


class PromptTemplate(AggregateRoot):
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
        super().__init__()
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
