from dataclasses import dataclass


@dataclass(frozen=True)
class ActivatePromptCommand:
    name: str
    version: int
    user_id: str | None = None
