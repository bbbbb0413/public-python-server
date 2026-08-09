from dataclasses import dataclass, field


@dataclass(frozen=True)
class CreatePromptCommand:
    name: str
    content: str
    variables: list[str] = field(default_factory=list)
    user_id: str | None = None
